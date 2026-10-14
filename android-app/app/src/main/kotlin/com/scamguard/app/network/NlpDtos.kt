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
    @SerializedName("scam_probability") val scamProbability: Double,
    @SerializedName("risk_level") val riskLevel: String,
    @SerializedName("patterns") val patterns: ScamPattern,
    @SerializedName("explanation") val explanation: String,
    @SerializedName("model") val model: String,
) {
    // Backward compatibility for existing code
    val scam_probability: Double get() = scamProbability
    val risk_level: String get() = riskLevel
}