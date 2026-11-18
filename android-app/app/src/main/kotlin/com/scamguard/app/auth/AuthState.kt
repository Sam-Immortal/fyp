package com.scamguard.app.auth

import android.content.Context
import android.content.SharedPreferences
import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Process-wide auth state.
 *
 * Tokens are persisted in private SharedPreferences because [com.scamguard.app.receiver.SmsReceiver]
 * can run when the app's UI process is not alive. Call [init] before reading any state
 * (MainActivity and SmsReceiver both do; it is idempotent).
 */
object AuthState {
    private const val PREFS_NAME = "auth_prefs"
    private const val KEY_ACCESS_TOKEN = "access_token"
    private const val KEY_REFRESH_TOKEN = "refresh_token"
    private const val KEY_NOTIFIED_EVENT_IDS = "notified_event_ids"
    private const val MAX_NOTIFIED_IDS = 200

    private val gson = Gson()

    @Volatile
    private var prefs: SharedPreferences? = null

    private val _accessToken = MutableStateFlow<String?>(null)
    val accessToken: StateFlow<String?> = _accessToken.asStateFlow()

    private val _refreshToken = MutableStateFlow<String?>(null)
    val refreshToken: StateFlow<String?> = _refreshToken.asStateFlow()

    private val _notifiedEventIds = MutableStateFlow<Set<Int>>(emptySet())
    val notifiedEventIds: StateFlow<Set<Int>> = _notifiedEventIds.asStateFlow()

    val isLoggedIn: Boolean
        get() = !_accessToken.value.isNullOrEmpty()

    @Synchronized
    fun init(applicationContext: Context) {
        if (prefs != null) return
        val p = applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs = p
        _accessToken.value = p.getString(KEY_ACCESS_TOKEN, null)
        _refreshToken.value = p.getString(KEY_REFRESH_TOKEN, null)
        _notifiedEventIds.value = loadNotifiedEventIds(p)
    }

    fun setTokens(accessToken: String, refreshToken: String) {
        _refreshToken.value = refreshToken
        _accessToken.value = accessToken
        prefs?.edit()
            ?.putString(KEY_ACCESS_TOKEN, accessToken)
            ?.putString(KEY_REFRESH_TOKEN, refreshToken)
            ?.apply()
    }

    fun clear() {
        _accessToken.value = null
        _refreshToken.value = null
        prefs?.edit()
            ?.remove(KEY_ACCESS_TOKEN)
            ?.remove(KEY_REFRESH_TOKEN)
            ?.apply()
    }

    @Synchronized
    fun markEventNotified(eventId: Int) {
        val updated = (_notifiedEventIds.value + eventId)
            .toList()
            .takeLast(MAX_NOTIFIED_IDS)
            .toSet()
        _notifiedEventIds.value = updated
        prefs?.edit()?.putString(KEY_NOTIFIED_EVENT_IDS, gson.toJson(updated))?.apply()
    }

    fun isEventNotified(eventId: Int): Boolean = eventId in _notifiedEventIds.value

    private fun loadNotifiedEventIds(p: SharedPreferences): Set<Int> {
        val json = p.getString(KEY_NOTIFIED_EVENT_IDS, null) ?: return emptySet()
        return try {
            gson.fromJson(json, Array<Int>::class.java)?.toSet() ?: emptySet()
        } catch (e: JsonSyntaxException) {
            emptySet()
        }
    }
}
