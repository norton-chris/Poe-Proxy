import asyncio
import fastapi_poe as fp
import os
async def list_available_bots():
    token = os.getenv("POE_API_KEY") or os.getenv("POE_TOKEN")
    if not token:
        print("Error: POE_API_KEY or POE_TOKEN environment variable not set")
        return
        
    try:
        client = await fp.get_client(token)
        bots = await client.get_bot_names()
        print("Available bots:")
        for bot in bots:
            print(f"- {bot}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(list_available_bots())
