package com.scamguard.app.network

import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.Headers
import retrofit2.http.POST

interface ScamGuardApi {
    @Headers("Content-Type: application/json")
    @POST("/auth/login")
    fun login(
        @Body request: LoginRequest
    ): Call<LoginResponse>

    @Headers("Content-Type: application/json")
    @POST("/auth/refresh")
    fun refresh(
        @Body request: RefreshRequest
    ): Call<RefreshResponse>

    @Headers("Content-Type: application/json")
    @POST("/auth/logout")
    fun logout(
        @Body request: LogoutRequest
    ): Call<ResponseBody>

    @Headers("Content-Type: application/json")
    @POST("/api/v1/nlp/score")
    fun scoreSms(
        @Body request: NlpScoreRequest
    ): Call<NlpScoreResponse>
}