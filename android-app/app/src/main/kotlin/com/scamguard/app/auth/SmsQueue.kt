package com.scamguard.app.auth

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import com.google.gson.reflect.TypeToken
import com.scamguard.app.network.NlpScoreRequest
import com.scamguard.app.network.ScamGuardApi
import java.io.IOException

/** An SMS waiting to be sent to the backend. */
data class PendingScan(
    val text: String,
    val sender: String,
    val timestamp: String,
)

/**
 * Persistent queue (SharedPreferences, max 50) for SMS that could not be submitted yet
 * (offline, not logged in, expired session). The SMS receiver adds to it; the UI drains it.
 *
 * All network methods are blocking: call them off the main thread.
 */
object SmsQueue {
    enum class Outcome { SENT, RETRY, DROP }

    private const val TAG = "SmsQueue"
    private const val PREFS = "sms_queue"
    private const val KEY = "queue"
    private const val MAX_QUEUED = 50

    private val gson = Gson()
    private val listType = object : TypeToken<List<PendingScan>>() {}.type
    private val lock = Any()
    private val drainLock = Any()

    /** Submit one SMS. Never logs message text. */
    fun submit(api: ScamGuardApi, scan: PendingScan): Outcome {
        if (!AuthState.isLoggedIn) return Outcome.RETRY
        return try {
            val request = NlpScoreRequest(
                text = scan.text,
                sender = scan.sender,
                timestamp = scan.timestamp,
            )
            val response = api.scoreSms(request).execute()
            val code = response.code()
            when {
                response.isSuccessful -> Outcome.SENT
                code == 401 || code == 429 || code >= 500 -> {
                    Log.w(TAG, "Submit failed with HTTP $code, will retry")
                    Outcome.RETRY
                }
                else -> {
                    Log.w(TAG, "Submit rejected with HTTP $code, dropping")
                    Outcome.DROP
                }
            }
        } catch (e: IOException) {
            Outcome.RETRY
        }
    }

    fun add(context: Context, scan: PendingScan) {
        synchronized(lock) {
            write(context, (read(context) + scan).takeLast(MAX_QUEUED))
        }
    }

    /** Try to send everything in the queue. Stops at the first retryable failure. */
    fun drain(context: Context, api: ScamGuardApi) {
        if (!AuthState.isLoggedIn) return
        synchronized(drainLock) {
            val snapshot = synchronized(lock) { read(context) }
            if (snapshot.isEmpty()) return

            val keep = mutableListOf<PendingScan>()
            for ((index, scan) in snapshot.withIndex()) {
                if (submit(api, scan) == Outcome.RETRY) {
                    keep.addAll(snapshot.drop(index))
                    break
                }
            }

            synchronized(lock) {
                // Anything appended while we were sending sits after our snapshot.
                val appended = read(context).drop(snapshot.size)
                write(context, (keep + appended).takeLast(MAX_QUEUED))
            }
        }
    }

    private fun read(context: Context): List<PendingScan> {
        val json = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY, null) ?: return emptyList()
        return try {
            gson.fromJson<List<PendingScan>>(json, listType) ?: emptyList()
        } catch (e: JsonSyntaxException) {
            emptyList()
        }
    }

    private fun write(context: Context, list: List<PendingScan>) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY, gson.toJson(list))
            .apply()
    }
}
