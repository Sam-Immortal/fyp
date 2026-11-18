package com.scamguard.app.network

import com.google.gson.annotations.SerializedName

data class NlpScoreRequest(
    @SerializedName("text") val text: String,
    @SerializedName("sender") val sender: String? = null,
    @SerializedName("timestamp") val timestamp: String? = null,
)

data class ScamPattern(
    @SerializedName("urgency") val urgency: Boolean = false,
    @SerializedName("authority_impersonation") val authorityImpersonation: Boolean = false,
    @SerializedName("credential_request") val credentialRequest: Boolean = false,
    @SerializedName("threat") val threat: Boolean = false,
    @SerializedName("reward") val reward: Boolean = false,
)

data class NlpScoreResponse(
    @SerializedName("event_id") val eventId: Int,
    @SerializedName("status") val status: String,
    @SerializedName("scam_probability") val scamProbability: Double,
    @SerializedName("risk_level") val riskLevel: String,
    @SerializedName("patterns") val patterns: ScamPattern,
    @SerializedName("explanation") val explanation: String,
    @SerializedName("model") val model: String,
)

data class NlpScoreEventResponse(
    @SerializedName("id") val id: Int,
    @SerializedName("status") val status: String,
    @SerializedName("sender") val sender: String,
    @SerializedName("message_preview") val messagePreview: String,
    @SerializedName("score") val score: Double,
    @SerializedName("threshold") val threshold: Double,
    @SerializedName("is_high_risk") val isHighRisk: Boolean,
    @SerializedName("risk_level") val riskLevel: String? = null,
    @SerializedName("patterns") val patterns: ScamPattern? = null,
    @SerializedName("explanation") val explanation: String? = null,
    @SerializedName("error") val error: String? = null,
    @SerializedName("created_at") val createdAt: String,
)