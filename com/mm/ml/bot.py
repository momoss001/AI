from wxpy import *
#初始化机器人
bot = Bot()

api_key = 'e81dbe7b23cf4d278a1d7193eaa833b8'
my_friend = ensure_one(bot.search('莫莫'))

tuling = Tuling(api_key=api_key)

@bot.register(my_friend  ,except_self=False)
def reply_my_friend(msg):
    tuling.do_reply(msg)

bot.join()

embed()
