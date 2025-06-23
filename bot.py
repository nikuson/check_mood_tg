import asyncio
import logging
import csv
import os
from datetime import datetime, timezone
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from transformers import pipeline
import pandas as pd
from config import BOT_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# грузим модель - долго но один раз
try:
    model = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
        return_all_scores=True
    )
    print("модель готова")
except Exception as e:
    print(f"модель не загрузилась: {e}")
    model = None

def create_data_folder():
    if not os.path.exists('data'):
        os.makedirs('data')

def save_to_csv(user_id, text, result, probs):
    create_data_folder()
    
    csv_path = 'data/requests.csv'
    is_new_file = not os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if is_new_file:
            # заголовки
            writer.writerow([
                'timestamp', 'user_id', 'text', 'sentiment', 
                'positive_prob', 'negative_prob', 'neutral_prob'
            ])
        
        # обрезаем длинный текст
        short_text = text[:500] if len(text) > 500 else text
        
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            user_id,
            short_text,
            result,
            probs.get('positive', 0),
            probs.get('negative', 0),
            probs.get('neutral', 0)
        ])

def check_mood(text):
    if not model:
        return None, {}
    
    try:
        results = model(text)[0]
        
        probs = {}
        mood = "нейтральный"
        best_score = 0
        
        for r in results:
            label = r['label'].lower()
            score = r['score']
            
            # TODO: переписать это нормально
            if 'positive' in label or 'pos' in label:
                probs['positive'] = round(score * 100, 1)
                if score > best_score:
                    best_score = score
                    mood = "позитивный"
            elif 'negative' in label or 'neg' in label:
                probs['negative'] = round(score * 100, 1) 
                if score > best_score:
                    best_score = score
                    mood = "негативный"
            else:
                probs['neutral'] = round(score * 100, 1)
                if score > best_score:
                    best_score = score
                    mood = "нейтральный"
        
        # заполняем пропущенные
        for category in ['positive', 'negative', 'neutral']:
            if category not in probs:
                probs[category] = 0.0
                
        return mood, probs
        
    except Exception as e:
        print(f"ошибка анализа: {e}")
        return None, {}

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Пиши любой текст - я скажу какое у него настроение 😊\n\n"
        "Команды:\n"
        "/stats - посмотреть статистику"
    )

@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    try:
        csv_path = 'data/requests.csv'
        if not os.path.exists(csv_path):
            await message.answer("Статистики пока нет, никто ничего не писал")
            return
            
        df = pd.read_csv(csv_path)
        
        total = len(df)
        
        # считаем проценты по настроениям
        mood_counts = df['sentiment'].value_counts()
        mood_percent = (mood_counts / total * 100).round(1)
        
        # средняя длина
        avg_len = df['text'].str.len().mean()
        
        response = f"📊 Статистика:\n\n"
        response += f"📝 Всего сообщений: {total}\n\n"
        response += f"📈 По настроениям:\n"
        
        for mood, percent in mood_percent.items():
            response += f"• {mood}: {percent}%\n"
            
        response += f"\n📏 Средняя длина: {avg_len:.0f} символов"
        
        await message.answer(response)
        
    except Exception as e:
        print(f"ошибка статистики: {e}")
        await message.answer("Что-то пошло не так со статистикой")

@dp.message(F.text)
async def handle_text(message: types.Message):
    if not model:
        await message.answer("Модель не работает, попробуй позже")
        return
    
    text = message.text
    user_id = message.from_user.id
    
    # анализируем
    mood, probs = check_mood(text)
    
    if mood is None:
        await message.answer("Не смог проанализировать текст")
        return
    
    # сохраняем в csv
    save_to_csv(user_id, text, mood, probs)
    
    # выбираем эмодзи
    if mood == "позитивный":
        emoji = "😊"
    elif mood == "негативный":
        emoji = "😞" 
    else:
        emoji = "😐"
    
    # отвечаем
    answer = f"{emoji} Настроение: **{mood}**\n\n"
    answer += "Вероятности:\n"
    answer += f"• Позитив: {probs['positive']}%\n"
    answer += f"• Негатив: {probs['negative']}%\n"
    answer += f"• Нейтрал: {probs['neutral']}%"
    
    await message.answer(answer, parse_mode="Markdown")

async def main():
    print("Запускаем бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())