package com.scamguard.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.scamguard.app.R
import com.scamguard.app.audio.AudioStreamingManager
import com.scamguard.app.auth.AuthState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun scamGuardApp(viewModel: AlertViewModel = viewModel()) {
    val authState = AuthState.accessToken.collectAsState()
    val isLoggedIn = authState.value != null

    // Streaming state
    val audioStreamingManager = remember { AudioStreamingManager(viewModel) }
    var isStreaming by remember { mutableStateOf(false) }
    var streamError by remember { mutableStateOf<String?>(null) }

    val state by viewModel.uiState.collectAsState()

    val backgroundColor = if (state.isHighRisk) Color(0xFFB00020) else MaterialTheme.colorScheme.background

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name)) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = if (state.isHighRisk) Color(0xFFB00020) else MaterialTheme.colorScheme.primaryContainer,
                ),
            )
        },
    ) { innerPadding ->
        if (!isLoggedIn) {
            // Show login screen
            LoginScreen(onLoginSuccess = { /* Triggers recomposition via AuthState */ })
        } else {
            // Show monitoring screen
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .background(backgroundColor),
                contentAlignment = Alignment.Center,
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    Text(
                        text = state.statusMessage,
                        style = MaterialTheme.typography.headlineMedium,
                        color = if (state.isHighRisk) Color.White else MaterialTheme.colorScheme.onBackground,
                    )
                    if (state.isHighRisk) {
                        Text(
                            text = stringResource(R.string.alert_high_risk),
                            style = MaterialTheme.typography.titleLarge,
                            color = Color.White,
                        )
                    }

                    androidx.compose.foundation.layout.Spacer(modifier = Modifier.size(32.dp))

                    // Start/Stop Monitoring Button
                    Card(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                        ),
                    ) {
                        Column(modifier = Modifier.padding(24.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Column {
                                    Text(
                                        text = if (isStreaming) stringResource(R.string.monitoring_active) else stringResource(R.string.monitoring_stopped),
                                        style = MaterialTheme.typography.titleMedium,
                                        fontWeight = FontWeight.Bold,
                                    )
                                    Text(
                                        text = if (isStreaming) stringResource(R.string.monitoring_description_active) else stringResource(R.string.monitoring_description_stopped),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                                Button(
                                    onClick = {
                                        if (isStreaming) {
                                            audioStreamingManager.stopStreaming()
                                            isStreaming = false
                                            streamError = null
                                        } else {
                                            streamError = null
                                            CoroutineScope(Dispatchers.IO).launch {
                                                audioStreamingManager.startStreaming()
                                                CoroutineScope(Dispatchers.Main).launch {
                                                    isStreaming = true
                                                }
                                            }
                                        }
                                    },
                                    enabled = !audioStreamingManager.isActive || isStreaming, // Allow stop even if not fully active
                                    colors = androidx.compose.material3.ButtonDefaults.buttonColors(
                                        containerColor = if (isStreaming) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                                        contentColor = if (isStreaming) MaterialTheme.colorScheme.onError else MaterialTheme.colorScheme.onPrimary,
                                    ),
                                ) {
                                    Text(
                                        text = if (isStreaming) stringResource(R.string.stop_monitoring) else stringResource(R.string.start_monitoring),
                                        style = MaterialTheme.typography.labelLarge,
                                    )
                                }
                            }

                            // Show streaming indicator
                            if (isStreaming) {
                                Row(
                                    modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
                                    horizontalArrangement = Arrangement.Center,
                                ) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(16.dp),
                                        color = MaterialTheme.colorScheme.primary,
                                    )
                                    androidx.compose.foundation.layout.Spacer(modifier = Modifier.padding(start = 8.dp))
                                    Text(
                                        stringResource(R.string.listening),
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.primary,
                                    )
                                }
                            }

                            streamError?.let { error ->
                                Text(
                                    text = "Error: $error",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.error,
                                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                                )
                            }
                        }
                    }

                    // Threshold info
                    Text(
                        text = stringResource(R.string.threshold_label, state.safetyThreshold),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun scamGuardAppPreview() {
    scamGuardApp()
}