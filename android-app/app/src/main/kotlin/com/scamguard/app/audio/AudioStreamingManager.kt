package com.scamguard.app.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import com.google.gson.Gson
import com.scamguard.app.BuildConfig
import com.scamguard.app.auth.AuthState
import com.scamguard.app.ui.AlertViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.nio.ByteBuffer
import kotlin.math.max
import java.util.UUID
import java.util.concurrent.TimeUnit

class AudioStreamingManager(
    private val alertViewModel: AlertViewModel,
) {
    private val TAG = "AudioStreamingManager"

    // Audio configuration matching backend expectations
    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

    private var audioRecord: AudioRecord? = null
    private var webSocket: WebSocket? = null
    private var webSocketJob: Job? = null
    private var audioRecordJob: Job? = null
    private var isStreaming = false

    private val gson = Gson()

    // WebSocket message types
    private data class ScoreMessage(
        val `type`: String,
        val chunk_index: Int,
        val anomaly_score: Double,
        val is_high_risk: Boolean,
        val threshold: Double,
        val processing_time_ms: Double,
        val timestamp: String,
    )

    fun startStreaming() {
        if (isStreaming) {
            Log.w(TAG, "Already streaming, ignoring start request")
            return
        }

        val token = AuthState.accessToken.value
        if (token == null || token.isBlank()) {
            Log.e(TAG, "No access token available, cannot start streaming")
            return
        }

        val sessionId = UUID.randomUUID().toString()
        val wsUrl = "${BuildConfig.WS_URL}/ws/audio?token=$token&session_id=$sessionId&sample_rate=16000"

        Log.i(TAG, "Starting audio streaming to $wsUrl")

        // Initialize AudioRecord
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT
        )
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "Invalid audio configuration")
            return
        }

        val bufferSize = max(minBufferSize, SAMPLE_RATE / 10) // ~100ms chunks
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        ).also {
            if (it.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                audioRecord = null
                return
            }
        }

        // Start WebSocket connection
        connectWebSocket(wsUrl)

        // Start audio recording loop
        isStreaming = true
        audioRecord?.startRecording()
        audioRecordJob = CoroutineScope(Dispatchers.IO).launch {
            recordAndStreamAudio()
        }
    }

    private fun connectWebSocket(url: String) {
        val okHttpClient = OkHttpClient.Builder()
            .pingInterval(10, TimeUnit.SECONDS)
            .build()

        val request = Request.Builder().url(url).build()

        webSocketJob = CoroutineScope(Dispatchers.IO).launch {
            okHttpClient.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                    Log.i(TAG, "WebSocket connected")
                    this@AudioStreamingManager.webSocket = webSocket
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    Log.d(TAG, "Received message: $text")
                    handleTextMessage(text)
                }

                override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                    // Server doesn't send binary frames
                }

                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                    Log.i(TAG, "WebSocket closing: $code $reason")
                    webSocket.close(1000, null)
                    this@AudioStreamingManager.webSocket = null
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    Log.i(TAG, "WebSocket closed: $code $reason")
                    this@AudioStreamingManager.webSocket = null
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
                    Log.e(TAG, "WebSocket failure", t)
                    this@AudioStreamingManager.webSocket = null
                }
            })
        }
    }

    private fun handleTextMessage(text: String) {
        try {
            val message = gson.fromJson(text, ScoreMessage::class.java)
            if (message.`type` == "score") {
                // Update UI on main thread
                CoroutineScope(Dispatchers.Main).launch {
                    alertViewModel.onAnomalyScore(message.anomaly_score.toFloat())
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse WebSocket message: $text", e)
        }
    }

    private fun recordAndStreamAudio() {
        val audioRecord = this.audioRecord ?: return
        val buffer = ShortArray(SAMPLE_RATE / 10) // 100ms buffer (1600 samples)

        while (isStreaming && !Thread.currentThread().isInterrupted) {
            val readResult = audioRecord.read(buffer, 0, buffer.size)
            if (readResult > 0) {
                // Convert to byte array and send
                val byteBuffer = ByteArray(readResult * 2)
                for (i in 0 until readResult) {
                    val sample = buffer[i]
                    byteBuffer[i * 2] = (sample.toInt() and 0xFF).toByte()
                    byteBuffer[i * 2 + 1] = (sample.toInt() shr 8).toByte()
                }
                sendAudioFrame(byteBuffer)
            } else if (readResult < 0) {
                Log.w(TAG, "AudioRecord read error: $readResult")
            }
            // Small yield to prevent busy loop
            Thread.yield()
        }
    }

    private fun sendAudioFrame(data: ByteArray) {
        val ws = webSocket
        if (ws != null && isStreaming) {
            try {
                ws.send(ByteString.of(*data))
            } catch (e: Exception) {
                Log.w(TAG, "Failed to send audio frame", e)
            }
        }
    }

    fun stopStreaming() {
        if (!isStreaming) {
            Log.w(TAG, "Not streaming, ignoring stop request")
            return
        }

        Log.i(TAG, "Stopping audio streaming")
        isStreaming = false

        // Capture and null out fields immediately to prevent double-tap races
        val jobToWait = audioRecordJob
        audioRecordJob = null
        val recordToRelease = audioRecord
        audioRecord = null

        // Close WebSocket (no race with AudioRecord)
        webSocket?.close(1000, "Client stopped")
        webSocket = null
        webSocketJob?.cancel()
        webSocketJob = null

        // Reset alert state on main thread
        CoroutineScope(Dispatchers.Main).launch {
            alertViewModel.reset()
        }

        // Wait for recording loop to exit, then safely stop/release AudioRecord
        // Must run on IO dispatcher since join() is a suspend function
        CoroutineScope(Dispatchers.IO).launch {
            jobToWait?.join()  // Wait for recordAndStreamAudio() to fully exit
            try {
                recordToRelease?.stop()
            } catch (e: Exception) {
                Log.w(TAG, "Error stopping AudioRecord", e)
            }
            try {
                recordToRelease?.release()
            } catch (e: Exception) {
                Log.w(TAG, "Error releasing AudioRecord", e)
            }
        }
    }

    val isActive: Boolean
        get() = isStreaming
}