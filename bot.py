import discord
import os

from nsfw_model.nsfw_detector import predict
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

client = discord.Client()
model = predict.load_model('./mobilenet_v2_140_224')

@client.event
async def on_ready():
    print(f"{client.user} has connected to Discord!")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    threshold = 0.5

    reasons = set()

    for attachment in message.attachments:
        url = attachment.url
        results = predict.classify(nsfw_model, url)
        results = list(results.values())[0]

        if results['drawings'] > threshold:
            reasons.add('Explicit Drawings')

        if results['hentai'] > threshold:
            reasons.add('Hentai')

        if results['porn'] > threshold:
            reasons.add('Pornography')

        if results['sexy'] > threshold:
            reasons.add('Sexual Imagery')

    if reasons:
        embed = discord.Embed(description = "**Reasons**: {reasons}".format(reasons = ', '.join(reasons)), colour = 0x7ecef2)
        embed.set_author(name = "{author}#{tag} has been censored (as a test)".format(author = message.author.display_name, tag = message.author.discriminator), icon_url = message.author.avatar_url)
        await message.delete()
        await message.channel.send(embed = embed)

client.run(TOKEN)
