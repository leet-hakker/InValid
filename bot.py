import discord
from discord.ext import commands
from discord.ext.commands import bot
from discord.utils import get
from discord import Member
import asyncio
import pandas as pd
from datetime import datetime
import pickle


command_prefix = "!"
client = commands.Bot(command_prefix)
classifier = pickle.load(open("trained_classifier.pickle", "rb"))


def extract_features(word_list):
    return dict([(word, True) for word in word_list])


def transform(message):
    return message.content


"""
@client.command(pass_context=True)
async def log(ctx):
    if ctx.author.id == 330404011197071360:
        pos = {}
        neg = {}
        await ctx.message.delete()
        for channel in ctx.guild.text_channels:
            print(channel.name + ": " + str(datetime.now()))
            if channel.id == 430970251174215690:
                history_pos = channel.history(limit=None)
                async for message in history_pos:
                    if not message.author.bot:
                        if message.author.id in pos:
                            pos[message.author.id] = (
                                pos[message.author.id] + message.content + " "
                            )
                        else:
                            pos[message.author.id] = message.content + " "
            else:
                history_neg = channel.history(limit=None)
                async for message in history_neg:
                    if not message.author.bot:
                        if message.author.id in neg:
                            neg[message.author.id] = (
                                neg[message.author.id] + message.content + " "
                            )
                        else:
                            neg[message.author.id] = message.content + " "

        print("Messages logged successfully\n")

        pos = list(pos.values())
        neg = list(neg.values())

        if len(pos) > len(neg):
            while len(pos) > len(neg):
                neg.append(None)
        elif len(neg) > len(pos):
            while len(neg) > len(pos):
                pos.append(None)

        df = pd.DataFrame({"pos": pos, "neg": neg})

        df.to_csv("messages.csv")
        print("Written to CSV\n")
"""


@client.event
async def on_message(ctx):
    probdist = classifier.prob_classify(extract_features(ctx.content.split()))
    pred_sentiment = probdist.max()
    print("Message: ", ctx.content)
    print("Sentiment: ", pred_sentiment)
    print("\n")


@client.event
async def on_ready():
    print("Ready\n\n")

client.run(TOKEN)
