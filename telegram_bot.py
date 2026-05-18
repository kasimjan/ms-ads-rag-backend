import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/chat")


def format_sources(sources):
    if not sources:
        return ""

    text = "\n\nSources:\n"

    for i, source in enumerate(sources, start=1):
        url = source.get("url", "")
        preview = source.get("preview", "")

        if url and url != "Unknown URL":
            text += f"{i}. {url}\n"
        else:
            text += f"{i}. Source unavailable\n"

        if preview:
            text += f"Preview: {preview[:180]}...\n"

    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "Hello! I am the UChicago MS ADS Assistant.\n\n"
        "Ask me questions about the MS in Applied Data Science program, "
        "application process, courses, capstone, online/in-person format, "
        "faculty, or research."
    )
    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "Example questions:\n\n"
        "- What is the MS ADS program?\n"
        "- Is GRE required?\n"
        "- What are the core courses?\n"
        "- Is the program STEM OPT eligible?\n"
        "- What is the capstone project?\n"
        "- What is the difference between online and in-person?"
    )
    await update.message.reply_text(message)


async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()

    if not question:
        await update.message.reply_text("Please send a question.")
        return

    await update.message.reply_text("Thinking...")

    try:
        payload = {
            "question": question,
            "k": 3
        }

        response = requests.post(BACKEND_URL, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()

        answer = data.get("answer", "I could not find an answer.")
        sources = data.get("sources", [])

        final_message = answer + format_sources(sources)

        if len(final_message) > 4000:
            final_message = final_message[:3900] + "\n\nMessage shortened because Telegram limit is 4096 characters."

        await update.message.reply_text(final_message)

    except requests.exceptions.ConnectionError:
        await update.message.reply_text(
            "Backend is not running. Start FastAPI first:\n\n"
            "uvicorn main:app --reload"
        )

    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. Set it before running the bot."
        )

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_question))

    print("Telegram bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()