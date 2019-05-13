import logging
from websocket_server import WebsocketServer
import sys

# reload(sys)
# sys.setdefaultencoding('utf-8')


def new_client(client, server):
    print("Client(%d) has joined." % client['id'])


def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])


def message_back(client, server, message):
    print("Client(%d) said: %s" % (client['id'], message))
    result = handle_msg(message)
    server.send_message(client, result)


def handle_msg(msg):
    res_msg = "ok"
    return res_msg


server = WebsocketServer(1024, host='192.168.0.128', loglevel=logging.INFO)

server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
server.set_fn_message_received(message_back)
server.run_forever()
