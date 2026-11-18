package com.scamguard.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.scamguard.app.auth.AuthState
import com.scamguard.app.ui.scamGuardApp

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Load persisted tokens before any UI or network call.
        AuthState.init(applicationContext)
        setContent {
            scamGuardApp()
        }
    }
}
