from wxpy import *
#初始化机器人
bot = Bot()

my_friend = bot.friends().search('莫莫')[0]


@bot.register(my_friend)
def print_msg(msg):
    print(msg.text)



friend2 = bot.friends().search("望蓝")[0]


