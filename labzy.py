import re

point_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*\]'
content = "[2, 2, 4, 5]"

point_match = re.search(point_pattern, content)
if point_match:
    print("匹配结果:", point_match.group(0))
    print("第一个数字:", point_match.group(1))
    print("第二个数字:", point_match.group(2))
else:
    print("未匹配到结果")