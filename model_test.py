#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# простой тест модели на реальных данных
from transformers import pipeline
import time

def test_model_accuracy():
    print("Загружаем модель...")
    
    try:
        model = pipeline(
            "sentiment-analysis",
            model="blanchefort/rubert-base-cased-sentiment",
            return_all_scores=True
        )
        print("✓ Модель загружена")
    except Exception as e:
        print(f"✗ Ошибка загрузки: {e}")
        return
    
    # тестовые фразы с ожидаемыми результатами
    test_cases = [
        ("Отличный день! Все замечательно!", "positive"),
        ("Ужасно плохо, ничего не получается", "negative"), 
        ("Сегодня дождь", "neutral"),
        ("Обожаю этот фильм! Супер!", "positive"),
        ("Кошмар какой-то, все сломалось", "negative"),
        ("Купил хлеб в магазине", "neutral"),
        ("Прекрасная погода сегодня", "positive"),
        ("Опять поломался компьютер", "negative")
    ]
    
    correct = 0
    total = len(test_cases)
    
    print(f"\nТестируем на {total} фразах...")
    print("-" * 50)
    
    for text, expected in test_cases:
        start = time.time()
        
        try:
            results = model(text)[0]
            
            # находим максимальную вероятность
            best_score = 0
            predicted = "neutral"
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if score > best_score:
                    best_score = score
                    if 'positive' in label or 'pos' in label:
                        predicted = "positive"
                    elif 'negative' in label or 'neg' in label:
                        predicted = "negative"
                    else:
                        predicted = "neutral"
            
            # проверяем правильность
            is_correct = predicted == expected
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            elapsed = time.time() - start
            
            print(f"{status} {text[:30]:30} | {expected:8} → {predicted:8} | {best_score:.2f} | {elapsed:.2f}s")
            
        except Exception as e:
            print(f"✗ Ошибка: {e}")
    
    # результаты
    accuracy = (correct / total) * 100
    print("-" * 50)
    print(f"Правильно: {correct}/{total}")
    print(f"Точность: {accuracy:.1f}%")
    
    if accuracy >= 85:
        print("✓ Модель прошла тест (>85%)")
    else:
        print("✗ Модель не прошла тест (<85%)")

if __name__ == "__main__":
    test_model_accuracy()