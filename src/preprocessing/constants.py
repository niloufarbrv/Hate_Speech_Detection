import emoji
from pathlib import Path
import re

BASE_PATH = Path(__file__).parent.parent.parent.absolute()

EMOTICANS = [':-)', ':)', '(:', '(-:', ':))', '((:', ':-D', ':D', 'X-D', 'XD', 'xD', 'xD', '<3', '</3', ':\*',
                ';-)',
                ';)', ';-D', ';D', '(;', '(-;', ':-(', ':(', '(:', '(-:', ':,(', ':\'(', ':"(', ':((', ':D', '=D',
                '=)',
                '(=', '=(', ')=', '=-O', 'O-=', ':o', 'o:', 'O:', 'O:', ':-o', 'o-:', ':P', ':p', ':S', ':s', ':@',
                ':>',
                ':<', '^_^', '^.^', '>.>', 'T_T', 'T-T', '-.-', '*.*', '~.~', ':*', ':-*', 'xP', 'XP', 'XP', 'Xp',
                ':-|',
                ':->', ':-<', '$_$', '8-)', ':-P', ':-p', '=P', '=p', ':*)', '*-*', 'B-)', 'O.o', 'X-(', ')-X']

URL_PATTERN = re.compile(r'https?://\S+')
USER_PATTERN = re.compile(r'@\w+\b')
NUMBER_PATTERN = re.compile(r'\d+')
EMOTICAN_PATTERN = {emotican: "<emoticon >" for emotican in EMOTICANS}
EMOJI_PATTERN = {char: "<emoticon >" for char in emoji.EMOJI_DATA}
HASHTAG_PATTERN = re.compile(r'#\w+')
