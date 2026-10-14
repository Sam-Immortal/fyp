package com.scamguard.app.service

import android.telecom.Call
import android.telecom.CallScreeningService

/**
 * Stub [CallScreeningService] that the system binds to screen incoming calls.
 *
 * Live call-audio capture is not implemented: modern Android restricts raw
 * in-call audio access to the default dialer / signature-level system apps
 * (CAPTURE_AUDIO_OUTPUT), which this app cannot hold. This class exists so
 * the manifest-declared <service> resolves to a real class; without it,
 * incoming calls would crash with ClassNotFoundException the moment the
 * system tries to bind this service.
 */
class CallScreeningService : CallScreeningService() {
    override fun onScreenCall(callDetails: Call.Details) {
        respondToCall(callDetails, CallResponse.Builder().build())
    }
}
