import unittest
import os
import csv
from unittest.mock import patch, mock_open
from bot import check_mood, save_to_csv, create_data_folder

class TestBot(unittest.TestCase):
    
    def test_create_data_folder(self):
        # проверяем что папка создается
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_mkdir:
            create_data_folder()
            mock_mkdir.assert_called_once_with('data')
    
    def test_check_mood_without_model(self):
        # если модель не загружена
        with patch('bot.model', None):
            mood, probs = check_mood("тестовый текст")
            self.assertIsNone(mood)
            self.assertEqual(probs, {})
    
    @patch('bot.model')
    def test_check_mood_positive(self, mock_model):
        # мокаем позитивный результат
        mock_model.return_value = [[
            {'label': 'POSITIVE', 'score': 0.9},
            {'label': 'NEGATIVE', 'score': 0.05},
            {'label': 'NEUTRAL', 'score': 0.05}
        ]]
        
        mood, probs = check_mood("отличный день!")
        
        self.assertEqual(mood, "позитивный")
        self.assertEqual(probs['positive'], 90.0)
        self.assertEqual(probs['negative'], 5.0)
        self.assertEqual(probs['neutral'], 5.0)
    
    @patch('bot.model')  
    def test_check_mood_negative(self, mock_model):
        # мокаем негативный результат
        mock_model.return_value = [[
            {'label': 'POSITIVE', 'score': 0.1},
            {'label': 'NEGATIVE', 'score': 0.8},
            {'label': 'NEUTRAL', 'score': 0.1}
        ]]
        
        mood, probs = check_mood("все плохо")
        
        self.assertEqual(mood, "негативный")
        self.assertEqual(probs['negative'], 80.0)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.isfile', return_value=True)
    @patch('bot.create_data_folder')
    def test_save_to_csv(self, mock_create_folder, mock_isfile, mock_file):
        # тестируем сохранение в CSV
        probs = {'positive': 10.0, 'negative': 80.0, 'neutral': 10.0}
        
        save_to_csv(123, "тест", "негативный", probs)
        
        mock_create_folder.assert_called_once()
        mock_file.assert_called_once()
        
        # проверяем что что-то записалось
        handle = mock_file()
        self.assertTrue(handle.write.called)

if __name__ == '__main__':
    # запускаем тесты
    unittest.main()