# Telegram Bot

A [Telegram bot](https://core.telegram.org/api#bot-api) is used to receive mobile notifications.


```julia
using Telegram, Telegram.API;
tg = TelegramClient("5507705604:AAHiYwaiILMx7PjZ2canJdp8ApUOb4DWvQw"; chat_id = "27807904");
```

The flyonic telegram bot token is: ```5507705604:AAHiYwaiILMx7PjZ2canJdp8ApUOb4DWvQw``` (Secret - do not share)

The telegram chat ID is user specific. To find it out:
1. open your Telegram app
2. search for ```@RawDataBot```
3. write the bot ```/start```
4. the bot will send you back your chat ID

To send a message with Julia simply write:
```julia
sendMessage(text = "Your message");
```