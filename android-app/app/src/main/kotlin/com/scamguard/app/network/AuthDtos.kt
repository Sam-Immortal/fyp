package com.scamguard.app.network

import com.google.gson.annotations.SerializedName

data class LoginRequest(
    @SerializedName("email") val email: String,
    @SerializedName("password") val password: String,
)

data class LoginResponse(
    @SerializedName("access_token") val accessToken: String,
    @SerializedName("refresh_token") val refreshToken: String,
    @SerializedName("token_type") val tokenType: String,
    @SerializedName("expires_in") val expiresIn: Int,
) {
    // Backward compatibility for existing code
    val access_token: String get() = accessToken
    val refresh_token: String get() = refreshToken
}

data class RefreshRequest(
    @SerializedName("refresh_token") val refreshToken: String,
)

data class RefreshResponse(
    @SerializedName("access_token") val accessToken: String,
    @SerializedName("refresh_token") val refreshToken: String,
    @SerializedName("token_type") val tokenType: String,
    @SerializedName("expires_in") val expiresIn: Int,
) {
    val access_token: String get() = accessToken
    val refresh_token: String get() = refreshToken
}

data class LogoutRequest(
    @SerializedName("refresh_token") val refreshToken: String,
)