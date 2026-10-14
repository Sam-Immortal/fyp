package com.scamguard.app

import com.scamguard.app.ui.AlertViewModel
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AlertViewModelTest {

    @Test
    fun `initial state is not high risk`() {
        val viewModel = AlertViewModel()
        val state = viewModel.uiState.value
        assertFalse(state.isHighRisk)
        assertEquals("ScamGuard is monitoring", state.statusMessage)
    }

    @Test
    fun `anomaly score above threshold flips to high risk`() {
        val viewModel = AlertViewModel()
        val threshold = viewModel.uiState.value.safetyThreshold

        viewModel.onAnomalyScore(threshold + 0.1f)

        val state = viewModel.uiState.value
        assertTrue(state.isHighRisk)
        assertTrue(state.statusMessage.contains("Anomaly score"))
    }

    @Test
    fun `anomaly score below threshold keeps monitoring state`() {
        val viewModel = AlertViewModel()
        val threshold = viewModel.uiState.value.safetyThreshold

        viewModel.onAnomalyScore(threshold - 0.1f)

        val state = viewModel.uiState.value
        assertFalse(state.isHighRisk)
        assertEquals("ScamGuard is monitoring", state.statusMessage)
    }

    @Test
    fun `reset returns to initial state`() {
        val viewModel = AlertViewModel()
        viewModel.onAnomalyScore(0.99f)
        viewModel.reset()

        val state = viewModel.uiState.value
        assertFalse(state.isHighRisk)
        assertEquals("ScamGuard is monitoring", state.statusMessage)
    }

    @Test
    fun `mockk verification example`() {
        val collaborator = mockk<Collaborator>(relaxed = true)
        every { collaborator.greet() } returns "hello"
        assertEquals("hello", collaborator.greet())
        verify { collaborator.greet() }
    }
}

interface Collaborator {
    fun greet(): String
}