package com.scamguard.app.ui

import android.Manifest
import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.repeatOnLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.scamguard.app.R
import com.scamguard.app.audio.AudioStreamingManager
import com.scamguard.app.auth.AuthState
import com.scamguard.app.auth.SmsQueue
import com.scamguard.app.network.ApiClient
import com.scamguard.app.network.NlpScoreEventResponse
import com.scamguard.app.network.ScamGuardApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

private const val ALERT_CHANNEL_ID = "high_risk_alerts"
private const val POLL_INTERVAL_MS = 3000L

// Fixed light card colours with dark text, so rows stay readable in light and dark theme.
private val TextPrimary = Color(0xFF1B1B1B)
private val TextSecondary = Color(0xFF444444)
private val DangerRed = Color(0xFFB00020)
private val SafeGreen = Color(0xFF2E7D32)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun scamGuardApp(viewModel: AlertViewModel = viewModel()) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    val accessToken by AuthState.accessToken.collectAsState()
    val isLoggedIn = !accessToken.isNullOrEmpty()
    val state by viewModel.uiState.collectAsState()

    // ---- Live audio monitoring -------------------------------------------------------------
    val audioStreamingManager = remember { AudioStreamingManager(viewModel) }
    var isStreaming by remember { mutableStateOf(false) }
    var streamError by remember { mutableStateOf<String?>(null) }

    // Declared before first use (Kotlin local functions must precede their callers).
    fun startStreaming() {
        scope.launch(Dispatchers.IO) {
            try {
                audioStreamingManager.startStreaming()
                withContext(Dispatchers.Main) { isStreaming = true }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    streamError = e.message ?: "Could not start monitoring"
                }
            }
        }
    }

    // ---- Permissions (requested from the screen that needs them) ----------------------------
    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) startStreaming() else streamError = "Microphone permission required"
    }

    var smsGranted by remember { mutableStateOf(hasPermission(context, Manifest.permission.RECEIVE_SMS)) }
    var notificationsGranted by remember { mutableStateOf(notificationsAllowed(context)) }

    val smsLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> smsGranted = granted }

    val notificationsLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> notificationsGranted = granted }

    // ---- SMS scans: the server is the source of truth ---------------------------------------
    val smsApi = remember { ApiClient.createApi() }
    var scans by remember { mutableStateOf<List<NlpScoreEventResponse>>(emptyList()) }
    var loadError by remember { mutableStateOf<String?>(null) }

    val lifecycleOwner = LocalLifecycleOwner.current
    LaunchedEffect(isLoggedIn) {
        if (!isLoggedIn) return@LaunchedEffect
        // Polls only while the screen is visible (STARTED); stops in the background.
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            while (true) {
                smsGranted = hasPermission(context, Manifest.permission.RECEIVE_SMS)
                notificationsGranted = notificationsAllowed(context)

                // Re-send anything the receiver could not submit earlier.
                withContext(Dispatchers.IO) { SmsQueue.drain(context.applicationContext, smsApi) }

                val (events, error) = fetchEvents(smsApi)
                if (events != null) {
                    scans = events.sortedByDescending { it.id }
                    loadError = null
                    notifyHighRisk(context, events)
                } else {
                    loadError = error
                }
                delay(POLL_INTERVAL_MS)
            }
        }
    }

    val backgroundColor = if (state.isHighRisk) DangerRed else MaterialTheme.colorScheme.background

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name)) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = if (state.isHighRisk) DangerRed else MaterialTheme.colorScheme.primaryContainer,
                ),
            )
        },
    ) { innerPadding ->
        if (!isLoggedIn) {
            LoginScreen(onLoginSuccess = { })
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .background(backgroundColor),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                item {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally,
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
                    }
                }

                item {
                    AudioCard(
                        isStreaming = isStreaming,
                        streamError = streamError,
                        onToggle = {
                            if (isStreaming) {
                                audioStreamingManager.stopStreaming()
                                isStreaming = false
                                streamError = null
                            } else {
                                streamError = null
                                if (hasPermission(context, Manifest.permission.RECORD_AUDIO)) {
                                    startStreaming()
                                } else {
                                    micLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                }
                            }
                        },
                    )
                }

                item {
                    Text(
                        text = stringResource(R.string.threshold_label, state.safetyThreshold),
                        style = MaterialTheme.typography.bodySmall,
                        color = if (state.isHighRisk) Color.White else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                if (!smsGranted || !notificationsGranted) {
                    item {
                        PermissionCard(
                            smsGranted = smsGranted,
                            notificationsGranted = notificationsGranted,
                            onRequestSms = { smsLauncher.launch(Manifest.permission.RECEIVE_SMS) },
                            onRequestNotifications = {
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                    notificationsLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                                }
                            },
                        )
                    }
                }

                item {
                    Text(
                        text = stringResource(R.string.sms_scans_title),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = if (state.isHighRisk) Color.White else MaterialTheme.colorScheme.onBackground,
                    )
                }

                loadError?.let { message ->
                    item {
                        Text(
                            text = stringResource(R.string.scan_error_prefix) + message,
                            style = MaterialTheme.typography.bodySmall,
                            color = if (state.isHighRisk) Color.White else MaterialTheme.colorScheme.error,
                        )
                    }
                }

                if (scans.isEmpty()) {
                    item {
                        Text(
                            text = stringResource(R.string.scan_no_scans),
                            style = MaterialTheme.typography.bodyMedium,
                            color = if (state.isHighRisk) Color.White else MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                } else {
                    items(scans, key = { it.id }) { scan ->
                        ScanRow(scan = scan)
                    }
                }
            }
        }
    }
}

@Composable
private fun AudioCard(
    isStreaming: Boolean,
    streamError: String?,
    onToggle: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
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
                    onClick = onToggle,
                    colors = ButtonDefaults.buttonColors(
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

            if (isStreaming) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.size(8.dp))
                    Text(
                        text = stringResource(R.string.listening),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }

            streamError?.let { message ->
                Text(
                    text = stringResource(R.string.scan_error_prefix) + message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                )
            }
        }
    }
}

@Composable
private fun PermissionCard(
    smsGranted: Boolean,
    notificationsGranted: Boolean,
    onRequestSms: () -> Unit,
    onRequestNotifications: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (!smsGranted) {
                Text(
                    text = stringResource(R.string.sms_permission_rationale),
                    style = MaterialTheme.typography.bodyMedium,
                )
                Button(onClick = onRequestSms) {
                    Text(stringResource(R.string.sms_permission_button))
                }
            }
            if (!notificationsGranted) {
                Text(
                    text = "Enable notifications to be alerted about high-risk scam messages.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Button(onClick = onRequestNotifications) {
                    Text("Enable Notifications")
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
private fun ScanRow(scan: NlpScoreEventResponse) {
    val isPending = scan.status == "pending"
    val isCompleted = scan.status == "completed"
    val isFailed = scan.status == "failed"
    // The backend decides what counts as high risk (risk level or score over the threshold).
    val isHighRisk = isCompleted && scan.isHighRisk

    val container = when {
        isPending -> Color(0xFFEEEEEE)
        isHighRisk -> Color(0xFFFFEBEE)
        isCompleted -> Color(0xFFE8F5E9)
        else -> Color(0xFFF5F5F5)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = container, contentColor = TextPrimary),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = scan.sender,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = scan.messagePreview,
                        style = MaterialTheme.typography.bodyMedium,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Spacer(modifier = Modifier.size(8.dp))
                when {
                    isPending -> Row(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(12.dp), strokeWidth = 2.dp)
                        Text(
                            text = stringResource(R.string.scan_status_analysing),
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                    isHighRisk -> Text(
                        text = stringResource(R.string.scan_status_high_risk),
                        style = MaterialTheme.typography.labelLarge,
                        color = DangerRed,
                        fontWeight = FontWeight.Bold,
                    )
                    isCompleted -> Text(
                        text = stringResource(R.string.scan_status_low_risk),
                        style = MaterialTheme.typography.labelLarge,
                        color = SafeGreen,
                        fontWeight = FontWeight.Bold,
                    )
                    isFailed -> Text(
                        text = stringResource(R.string.scan_status_failed),
                        style = MaterialTheme.typography.labelLarge,
                        color = TextSecondary,
                    )
                }
            }

            if (isCompleted) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        text = stringResource(R.string.scan_scam_probability),
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                    )
                    Text(
                        text = "%.0f%%".format(scan.score * 100),
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = if (isHighRisk) DangerRed else SafeGreen,
                    )
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        text = stringResource(R.string.scan_risk_level),
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                    )
                    Text(
                        text = scan.riskLevel?.uppercase() ?: "UNKNOWN",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                    )
                }

                scan.explanation?.let { explanation ->
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = explanation,
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                    )
                }

                scan.patterns?.let { patterns ->
                    val detected = mutableListOf<String>()
                    if (patterns.urgency) detected.add("Urgency")
                    if (patterns.authorityImpersonation) detected.add("Authority impersonation")
                    if (patterns.credentialRequest) detected.add("Credential request")
                    if (patterns.threat) detected.add("Threat")
                    if (patterns.reward) detected.add("Reward")

                    if (detected.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = stringResource(R.string.scan_patterns),
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                        )
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalArrangement = Arrangement.spacedBy(0.dp),
                        ) {
                            detected.forEach { name ->
                                AssistChip(
                                    onClick = { },
                                    label = { Text(text = name, style = MaterialTheme.typography.labelSmall) },
                                )
                            }
                        }
                    }
                }
            }

            if (isFailed) {
                scan.error?.let { message ->
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = stringResource(R.string.scan_error_prefix) + message.take(120),
                        style = MaterialTheme.typography.bodySmall,
                        color = DangerRed,
                    )
                }
            }
        }
    }
}

// ---- Helpers (plain functions: no @Composable, no LocalContext) --------------------------------

private fun hasPermission(context: Context, permission: String): Boolean =
    ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED

private fun notificationsAllowed(context: Context): Boolean =
    Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
        hasPermission(context, Manifest.permission.POST_NOTIFICATIONS)

/** Returns (events, null) on success or (null, short error message) on failure. */
private suspend fun fetchEvents(api: ScamGuardApi): Pair<List<NlpScoreEventResponse>?, String?> =
    withContext(Dispatchers.IO) {
        try {
            val response = api.listNlpEvents(limit = 50).execute()
            if (response.isSuccessful) {
                Pair(response.body() ?: emptyList(), null)
            } else {
                Pair(null, "HTTP ${response.code()}")
            }
        } catch (e: IOException) {
            Pair(null, "Cannot reach server")
        }
    }

private fun ensureAlertChannel(context: Context) {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
    val manager = context.getSystemService(NotificationManager::class.java)
    manager.createNotificationChannel(
        NotificationChannel(ALERT_CHANNEL_ID, "High-risk scam alerts", NotificationManager.IMPORTANCE_HIGH),
    )
}

/** One notification per high-risk event id (remembered across restarts). */
@SuppressLint("MissingPermission") // guarded by notificationsAllowed()
private fun notifyHighRisk(context: Context, events: List<NlpScoreEventResponse>) {
    if (!notificationsAllowed(context)) return
    val fresh = events.filter {
        it.status == "completed" && it.isHighRisk && !AuthState.isEventNotified(it.id)
    }
    if (fresh.isEmpty()) return

    ensureAlertChannel(context)
    val manager = NotificationManagerCompat.from(context)
    for (event in fresh) {
        val notification = NotificationCompat.Builder(context, ALERT_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("Scam warning")
            .setContentText("From ${event.sender}: ${event.messagePreview}")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        manager.notify(event.id, notification)
        AuthState.markEventNotified(event.id)
    }
}

@Preview(showBackground = true)
@Composable
private fun scamGuardAppPreview() {
    scamGuardApp()
}
