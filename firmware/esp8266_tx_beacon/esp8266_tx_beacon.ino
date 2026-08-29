/*
 * ESP8266 CSI TX Beacon Firmware (Arduino IDE)
 * ============================================
 * Converts ESP8266 into a high-speed WiFi packet generator (TX Beacon)
 * for the ESP32-S3 CSI Fall Detection System.
 * 
 * INSTRUCTIONS FOR FLUSHING YOUR 6 ESP8266 UNITS:
 * 1. Open Arduino IDE.
 * 2. Install ESP8266 board support if not installed:
 *    - Preferences -> Additional Boards Manager URLs: 
 *      http://arduino.esp8266.com/stable/package_esp8266com_index.json
 *    - Tools -> Board -> Boards Manager -> Search "esp8266" -> Install
 * 3. Open this file in Arduino IDE.
 * 4. Change `NODE_ID` for each unit (1, 2, 3, 4, 5, or 6):
 *    - Unit 1: #define NODE_ID 1  (MAC: 1A:00:00:00:00:01)
 *    - Unit 2: #define NODE_ID 2  (MAC: 1A:00:00:00:00:02)
 *    - Unit 3: #define NODE_ID 3  (MAC: 1A:00:00:00:00:03)
 *    - Unit 4: #define NODE_ID 4  (MAC: 1A:00:00:00:00:04)
 *    - Unit 5: #define NODE_ID 5  (MAC: 1A:00:00:00:00:05)
 *    - Unit 6: #define NODE_ID 6  (MAC: 1A:00:00:00:00:06)
 * 5. Plug in ESP8266 via USB, select Board: "NodeMCU 1.0 (ESP-12E Module)" or "Generic ESP8266 Module".
 * 6. Click Upload.
 * 7. Unplug and move to its designated room corner with any USB power adapter!
 */

#include <ESP8266WiFi.h>
#include <espnow.h>

// ============================================================================
// CHANGE THIS ID FOR EACH OF YOUR 6 ESP8266 UNITS (1 to 6)
// ============================================================================
#define NODE_ID 6

#define WIFI_CHANNEL 6      // Must match ESP32-S3 receiver channel (Channel 6)
#define SEND_INTERVAL_MS 20 // 20ms = 50 packets/sec (HT20)

// ESP32-S3 Receiver 1 (COM3) MAC: AC:A7:04:2C:41:DC
uint8_t rx1Address[] = {0xAC, 0xA7, 0x04, 0x2C, 0x41, 0xDC};
// ESP32-S3 Receiver 2 (COM5) MAC: AC:A7:04:2C:2B:E8
uint8_t rx2Address[] = {0xAC, 0xA7, 0x04, 0x2C, 0x2B, 0xE8};

// Data payload structure
struct __attribute__((packed)) BeaconPayload {
    uint32_t node_id;
    uint32_t sequence;
    uint32_t uptime_ms;
};

BeaconPayload payload;
uint32_t seq_count = 0;

void setup() {
    Serial.begin(115200);
    Serial.printf("\n--- ESP8266 CSI TX Beacon #%d ---\n", NODE_ID);

    // 1. Configure WiFi
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    wifi_set_channel(WIFI_CHANNEL);
    wifi_set_phy_mode(PHY_MODE_11N);

    // 2. Set Custom MAC Address (1A:00:00:00:00:0X)
    uint8_t customMac[6] = {0x1A, 0x00, 0x00, 0x00, 0x00, (uint8_t)NODE_ID};
    wifi_set_macaddr(STATION_IF, customMac);

    Serial.print("MAC Address set to: ");
    Serial.println(WiFi.macAddress());
    Serial.printf("WiFi Channel: %d\n", WIFI_CHANNEL);

    // 3. Initialize ESP-NOW
    if (esp_now_init() != 0) {
        Serial.println("Error initializing ESP-NOW!");
        return;
    }

    esp_now_set_self_role(ESP_NOW_ROLE_CONTROLLER);
    // Add both receivers as peers
    esp_now_add_peer(rx1Address, ESP_NOW_ROLE_SLAVE, WIFI_CHANNEL, NULL, 0);
    esp_now_add_peer(rx2Address, ESP_NOW_ROLE_SLAVE, WIFI_CHANNEL, NULL, 0);

    Serial.println("ESP-NOW Unicast Transmitter initialized successfully!");
}

void loop() {
    payload.node_id = NODE_ID;
    payload.sequence = seq_count++;
    payload.uptime_ms = millis();

    // Send unicast packet via ESP-NOW to both receivers
    esp_now_send(rx1Address, (uint8_t *)&payload, sizeof(payload));
    esp_now_send(rx2Address, (uint8_t *)&payload, sizeof(payload));

    delay(SEND_INTERVAL_MS);
}
