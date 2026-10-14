package com.scamguard.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.scamguard.app.ui.scamGuardApp

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            PermissionGate(
                permission = Manifest.permission.RECORD_AUDIO,
                rationale = stringResource(R.string.permission_rationale),
            ) {
                scamGuardApp()
            }
        }
    }
}

@Composable
fun PermissionGate(
    permission: String,
    rationale: String,
    onPermissionGranted: @Composable () -> Unit,
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val permissionGranted = remember { mutableStateOf(false) }

    val requestPermissionLauncher = androidx.activity.compose.rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        permissionGranted.value = isGranted
    }

    // Check permission on composition
    androidx.compose.runtime.LaunchedEffect(Unit) {
        val granted = context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED
        permissionGranted.value = granted
        if (!granted) {
            // Check if we should show rationale
            val shouldShowRationale = androidx.core.app.ActivityCompat.shouldShowRequestPermissionRationale(context as android.app.Activity, permission)
            if (shouldShowRationale) {
                // The rationale will be shown in the UI below
            } else {
                requestPermissionLauncher.launch(permission)
            }
        }
    }

    if (permissionGranted.value) {
        onPermissionGranted()
    } else {
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = stringResource(R.string.permission_rationale),
                style = androidx.compose.material3.MaterialTheme.typography.headlineMedium,
                modifier = Modifier.padding(24.dp),
            )
            Text(
                text = rationale,
                style = androidx.compose.material3.MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(24.dp),
            )
            Button(
                onClick = { requestPermissionLauncher.launch(permission) },
                modifier = Modifier.padding(24.dp),
            ) {
                Text(stringResource(R.string.permission_button), style = androidx.compose.material3.MaterialTheme.typography.labelLarge)
            }
        }
    }
}