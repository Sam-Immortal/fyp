package com.scamguard.app.network

import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Headers
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

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

    @GET("/api/v1/nlp/events")
    fun listNlpEvents(
        @Query("limit") limit: Int = 50,
        @Query("offset") offset: Int = 0,
        @Query("since_id") sinceId: Long? = null,
    ): Call<List<NlpScoreEventResponse>>

    @GET("/api/v1/nlp/events/{eventId}")
    fun getNlpEvent(
        @Path("eventId") eventId: Long,
    ): Call<NlpScoreEventResponse>
}