package com.scamguard.app.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.telephony.SmsMessage
import android.util.Log
import com.scamguard.app.network.ApiClient
import com.scamguard.app.network.NlpScoreRequest
import com.scamguard.app.network.NlpScoreResponse
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

/**
 * [BroadcastReceiver] for incoming SMS messages.
 * Forwards the SMS payload to the FastAPI backend for NLP scam classification.
 */
class SmsReceiver : BroadcastReceiver() {
    private val TAG = "SmsReceiver"
    private val api = ApiClient.createApi()

    override fun onReceive(
        context: Context,
        intent: Intent,
    ) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        val bundle = intent.extras ?: return
        val pdus = bundle.get("pdus") as? Array<Any> ?: return

        val messages = pdus.mapNotNull { pdu ->
            (pdu as? ByteArray)?.let { SmsMessage.createFromPdu(it) }
        }

        for (message in messages) {
            val sender = message.originatingAddress ?: "unknown"
            val body = message.messageBody ?: ""
            val timestamp = System.currentTimeMillis().toString()

            Log.i(TAG, "Received SMS from $sender: $body")

            dispatchToBackend(sender, body, timestamp)
        }
    }

    private fun dispatchToBackend(
        sender: String,
        body: String,
        timestamp: String,
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            val request = NlpScoreRequest(
                text = body,
                sender = sender,
                timestamp = timestamp,
            )

            api.scoreSms(request).enqueue(
                object : Callback<NlpScoreResponse> {
                    override fun onResponse(
                        call: Call<NlpScoreResponse>,
                        response: Response<NlpScoreResponse>,
                    ) {
                        if (response.isSuccessful) {
                            Log.i(TAG, "NLP analysis result: ${response.body()}")
                        } else {
                            Log.w(TAG, "NLP analysis failed: ${response.code()} ${response.message()}")
                        }
                    }

                    override fun onFailure(
                        call: Call<NlpScoreResponse>,
                        t: Throwable,
                    ) {
                        Log.e(TAG, "NLP analysis error", t)
                    }
                },
            )
        }
    }
}
