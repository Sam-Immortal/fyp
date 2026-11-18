package com.scamguard.app.network

import com.google.gson.GsonBuilder
import com.scamguard.app.BuildConfig
import com.scamguard.app.auth.AuthState
import okhttp3.Authenticator
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock

object ApiClient {
    private const val TIMEOUT_SECONDS = 30L
    private const val SMS_CONNECT_TIMEOUT_SECONDS = 5L
    private const val SMS_READ_TIMEOUT_SECONDS = 8L

    // Endpoints where a 401 means "wrong credentials", not "expired access token".
    private val NO_REFRESH_PATHS = listOf("/auth/login", "/auth/register", "/auth/refresh", "/auth/logout")

    private val gson = GsonBuilder().setLenient().create()

    private fun Request.withBearer(token: String): Request =
        newBuilder().header("Authorization", "Bearer $token").build()

    // Adds the current Bearer token to every request.
    private val authInterceptor = Interceptor { chain ->
        val original = chain.request()
        val token = AuthState.accessToken.value
        if (token.isNullOrEmpty()) {
            chain.proceed(original)
        } else {
            chain.proceed(original.withBearer(token))
        }
    }

    private val refreshLock = ReentrantLock()

    // Separate client with NO authenticator and NO bearer header, so refreshing cannot loop.
    private val refreshApi: ScamGuardApi by lazy {
        val client = OkHttpClient.Builder()
            .connectTimeout(SMS_CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(SMS_READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .writeTimeout(SMS_READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .build()
        Retrofit.Builder()
            .baseUrl(BuildConfig.API_BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
            .create(ScamGuardApi::class.java)
    }

    /**
     * Handles 401 on normal API calls:
     * 1. never retry a request that is already a retry;
     * 2. if another request already refreshed the token, retry with the new one;
     * 3. otherwise refresh once, store BOTH new tokens, retry with the new access token;
     * 4. if the server rejects the refresh token, clear the session (user must log in again).
     *    A network error keeps the session so the app still works once back online.
     */
    private val authenticator = Authenticator { _, response ->
        if (response.priorResponse != null) return@Authenticator null
        val path = response.request.url.encodedPath
        if (NO_REFRESH_PATHS.any { path.endsWith(it) }) return@Authenticator null

        val failedHeader = response.request.header("Authorization")

        refreshLock.lock()
        try {
            val current = AuthState.accessToken.value
            if (!current.isNullOrEmpty() && failedHeader != "Bearer $current") {
                // Token changed since this request was built: just retry with the newer one.
                return@Authenticator response.request.withBearer(current)
            }

            val refreshToken = AuthState.refreshToken.value
            if (refreshToken.isNullOrEmpty()) {
                AuthState.clear()
                return@Authenticator null
            }

            try {
                val refreshResponse = refreshApi.refresh(RefreshRequest(refreshToken = refreshToken)).execute()
                val body = refreshResponse.body()
                if (refreshResponse.isSuccessful && body != null) {
                    AuthState.setTokens(body.accessToken, body.refreshToken)
                    return@Authenticator response.request.withBearer(body.accessToken)
                }
                if (refreshResponse.code() in 400..499) {
                    AuthState.clear()
                }
            } catch (e: IOException) {
                // Offline or server unreachable: keep the tokens, give up on this request.
            }
            null
        } finally {
            refreshLock.unlock()
        }
    }

    // BASIC level + redacted header: never log bodies (SMS text, tokens, passwords).
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = if (BuildConfig.DEBUG) HttpLoggingInterceptor.Level.BASIC else HttpLoggingInterceptor.Level.NONE
        redactHeader("Authorization")
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .addInterceptor(authInterceptor)
        .authenticator(authenticator)
        .connectTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .readTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .writeTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .build()

    // Short timeouts: used from the SMS receiver, which must finish quickly.
    private val smsOkHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .addInterceptor(authInterceptor)
        .authenticator(authenticator)
        .connectTimeout(SMS_CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .readTimeout(SMS_READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .writeTimeout(SMS_READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .build()

    private val retrofit = Retrofit.Builder()
        .baseUrl(BuildConfig.API_BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create(gson))
        .build()

    private val smsRetrofit = Retrofit.Builder()
        .baseUrl(BuildConfig.API_BASE_URL)
        .client(smsOkHttpClient)
        .addConverterFactory(GsonConverterFactory.create(gson))
        .build()

    fun createApi(): ScamGuardApi = retrofit.create(ScamGuardApi::class.java)

    fun createSmsApi(): ScamGuardApi = smsRetrofit.create(ScamGuardApi::class.java)
}
