from nrf24 import NRF24
import time

# pipes = [[0xe7, 0xe7, 0xe7, 0xe7, 0xe7], [0xc2, 0xc2, 0xc2, 0xc2, 0xc2]]
pipes = [[0x31, 0x4e, 0x6f, 0x64, 0x65], [0x32, 0x4e, 0x6f, 0x64, 0x65]]

radio = NRF24()
radio.begin(1, 0, "P8_23", "P8_24")

radio.setRetries(15,15)

radio.setPayloadSize(8)
radio.setChannel(0x60)
radio.setDataRate(NRF24.BR_250KBPS)
radio.setPALevel(NRF24.PA_MAX)

radio.setAutoAck(1)

radio.openWritingPipe(pipes[0])
radio.openReadingPipe(1, pipes[1])

radio.startListening()
radio.stopListening()

radio.printDetails()

radio.startListening()

while True:
    pipe = [0]
    while not radio.available(pipe, True):
        time.sleep(1000/1000000.0)

    recv_buffer = []
    radio.read(recv_buffer)

    print(recv_buffer)