package com.scamguard.app.ui

import androidx.lifecycle.ViewModel
import com.scamguard.app.BuildConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class AlertUiState(
    val isHighRisk: Boolean = false,
    val statusMessage: String = "ScamGuard is monitoring",
) {
    val safetyThreshold: Float = BuildConfig.SAFETY_THRESHOLD
}

class AlertViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(AlertUiState())
    val uiState: StateFlow<AlertUiState> = _uiState.asStateFlow()

    fun onAnomalyScore(score: Float) {
        val isHigh = score >= _uiState.value.safetyThreshold
        _uiState.update { current ->
            current.copy(
                isHighRisk = isHigh,
                statusMessage = if (isHigh) {
                    "Anomaly score: $score"
                } else {
                    "ScamGuard is monitoring"
                },
            )
        }
    }

    fun reset() {
        _uiState.value = AlertUiState()
    }
}