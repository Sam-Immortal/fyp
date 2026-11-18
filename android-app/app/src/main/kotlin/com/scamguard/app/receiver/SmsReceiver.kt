package com.scamguard.app.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log
import com.scamguard.app.auth.AuthState
import com.scamguard.app.auth.PendingScan
import com.scamguard.app.auth.SmsQueue
import com.scamguard.app.network.ApiClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.time.Instant

/**
 * Receives incoming SMS and submits them to the backend for scam classification.
 *
 * The backend answers 202 immediately; the verdict is read later by the UI from
 * GET /api/v1/nlp/events. Anything that cannot be submitted goes to [SmsQueue].
 * Message text is never logged.
 */
class SmsReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        val appContext = context.applicationContext
        // The process may have been started just for this broadcast: load persisted tokens.
        AuthState.init(appContext)

        val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)
        if (messages.isNullOrEmpty()) return

        // Merge multipart messages from the same sender into one text.
        val scans = messages
            .groupBy { it.originatingAddress ?: "unknown" }
            .map { (sender, parts) ->
                PendingScan(
                    text = parts.joinToString(separator = "") { it.messageBody ?: "" },
                    sender = sender,
                    timestamp = Instant.now().toString(), // ISO-8601 UTC
                )
            }
            .filter { it.text.isNotBlank() }
        if (scans.isEmpty()) return

        // goAsync() may only be called once per broadcast; finish() only after the work is done.
        val pendingResult = goAsync()
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val api = ApiClient.createSmsApi()
                for (scan in scans) {
                    val outcome = try {
                        SmsQueue.submit(api, scan)
                    } catch (e: Exception) {
                        SmsQueue.Outcome.RETRY
                    }
                    when (outcome) {
                        SmsQueue.Outcome.SENT ->
                            Log.i(TAG, "SMS submitted for analysis (length=${scan.text.length})")
                        SmsQueue.Outcome.RETRY -> {
                            SmsQueue.add(appContext, scan)
                            Log.w(TAG, "SMS queued for retry")
                        }
                        SmsQueue.Outcome.DROP ->
                            Log.w(TAG, "SMS rejected by server, not retried")
                    }
                }
            } finally {
                pendingResult.finish()
            }
        }
    }

    private companion object {
        const val TAG = "SmsReceiver"
    }
}
