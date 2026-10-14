package com.scamguard.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.scamguard.app.R
import com.scamguard.app.auth.AuthState
import com.scamguard.app.network.ApiClient
import com.scamguard.app.network.LoginRequest
import com.scamguard.app.network.LoginResponse
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LoginScreen(onLoginSuccess: () -> Unit) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }

    val api = ApiClient.createApi()

    // Get string resources outside the callback
    val loginFailedFormat = stringResource(R.string.login_failed)
    val networkErrorFormat = stringResource(R.string.network_error)

    val handleLogin = {
        errorMessage = null
        isLoading = true

        val request = LoginRequest(email, password)
        api.login(request).enqueue(object : Callback<LoginResponse> {
            override fun onResponse(
                call: Call<LoginResponse>,
                response: Response<LoginResponse>
            ) {
                isLoading = false
                if (response.isSuccessful && response.body() != null) {
                    val body = response.body()!!
                    AuthState.setTokens(body.access_token, body.refresh_token)
                    // AuthState change will trigger recomposition and show monitoring screen
                } else {
                    errorMessage = loginFailedFormat.format(response.code())
                }
            }

            override fun onFailure(call: Call<LoginResponse>, t: Throwable) {
                isLoading = false
                errorMessage = networkErrorFormat.format(t.message ?: "Unknown error")
            }
        })
    }

    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        androidx.compose.foundation.layout.Spacer(modifier = Modifier.size(32.dp))

        Text(
            text = stringResource(R.string.login_title),
            style = androidx.compose.material3.MaterialTheme.typography.headlineLarge,
            fontSize = 32.sp,
        )

        Text(
            text = stringResource(R.string.login_subtitle),
            style = androidx.compose.material3.MaterialTheme.typography.bodyLarge,
        )

        androidx.compose.foundation.layout.Spacer(modifier = Modifier.size(16.dp))

        OutlinedTextField(
            value = email,
            onValueChange = { email = it },
            label = { Text(stringResource(R.string.email_hint)) },
            modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp),
            singleLine = true,
        )

        OutlinedTextField(
            value = password,
            onValueChange = { password = it },
            label = { Text(stringResource(R.string.password_hint)) },
            modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp),
            visualTransformation = PasswordVisualTransformation(),
            singleLine = true,
        )

        errorMessage?.let { msg ->
            Text(
                text = msg,
                color = androidx.compose.material3.MaterialTheme.colorScheme.error,
                style = androidx.compose.material3.MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(horizontal = 24.dp),
            )
        }

        Button(
            onClick = handleLogin,
            enabled = !isLoading && email.isNotBlank() && password.isNotBlank(),
            modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp),
        ) {
            if (isLoading) {
                androidx.compose.material3.CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    color = androidx.compose.material3.MaterialTheme.colorScheme.onPrimary,
                )
            } else {
                Text(stringResource(R.string.login_button), style = androidx.compose.material3.MaterialTheme.typography.labelLarge)
            }
        }
    }
}

@Composable
@Preview(showBackground = true)
fun LoginScreenPreview() {
    LoginScreen(onLoginSuccess = {})
}