import asyncio

from client import DemoEnv
from models import DemoAction


async def main():
    env = DemoEnv("http://localhost:8000")

    obs = await env.reset()
    print(obs)

    for letter in ["a", "e", "i", "o", "u", "t", "r", "s"]:
        result = await env.step(DemoAction(message=letter))
        print(result.observation.echoed_message, result.observation.message_length)

        if result.done:
            break


asyncio.run(main())