package com.google.mediapipe.examples.handlandmarker

import android.util.Log
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit

object WebSocketClient : WebSocketListener() {

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS) // Keep the connection alive
        .build()

    private var webSocket: WebSocket? = null
    private val serverUrl = "ws://0.0.0.0:5000"

    private var listener: WebSocketMessageListener? = null
    private var isConnected = false

    fun connect() {
        if (isConnected) return // Prevent multiple connections

        val request = Request.Builder().url(serverUrl).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("WebSocket", "Connected to server")
                isConnected = true
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d("WebSocket", "Received: $text")
                listener?.onMessageReceived(text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("WebSocket", "Error: ${t.message}")
                isConnected = false
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d("WebSocket", "Closed: $reason")
                isConnected = false
            }
        })
    }

    fun sendLandmarks(leftHand: List<NormalizedLandmark>?, rightHand: List<NormalizedLandmark>?) {
        val leftData = formatLandmarks(leftHand)
        val rightData = formatLandmarks(rightHand)

        val jsonObject = JSONObject()
        jsonObject.put("landmarks", "$leftData,$rightData")

        try {
            webSocket?.send(jsonObject.toString())
            Log.d("WebSocket", "Sent: $jsonObject")
        } catch (e: Exception) {
            Log.e("WebSocket", "Send failed", e)
        }
    }

    private fun formatLandmarks(landmarks: List<NormalizedLandmark>?): String {
        return landmarks?.joinToString(",") { "${it.x()},${it.y()},${it.z()}" }
            ?: List(63) { "0.0" }.joinToString(",") // If null, return 63 zeros
    }

    fun close() {
        webSocket?.close(1000, "Closing connection")
        isConnected = false
    }

    fun setListener(newListener: WebSocketMessageListener) {
        listener = newListener
    }

    interface WebSocketMessageListener {
        fun onMessageReceived(message: String)
    }
}
