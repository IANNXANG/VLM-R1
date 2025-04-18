import re

pattern = r'.'
text = 'Hello\nWorld'
matches = re.findall(pattern, text, re.DOTALL)
print(matches)