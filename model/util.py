import re
from random import random

def cut_word(str):
    """
    这个逻辑和java部分里切词逻辑是一样的
    再写一遍 我是纯纯的冤种

    :param str: 要切词的字符串
    :return: 切好的token序列
    """
    parenthesesRegex = "(.*)\\((.*)\\)\Z"  # xxx(xxx)
    equalRegex = "(.*)=('.*')\Z"  # xxx='xxx'

    result = []

    str = str[1:-1]  # 去掉引号

    if re.match(parenthesesRegex, str):
        tokens = str.split('(')
        token1 = tokens[0][:-1]
        token2 = tokens[1][:-1]

        sub_tokens_1 = cut_hump(token1).split(' ')
        sub_tokens_2 = cut_hump(token2).split(' ')

        for sub_token in sub_tokens_1:
            result.append(sub_token)
        for sub_token in sub_tokens_2:
            result.append(sub_token)

    elif re.match(equalRegex, str):
        tokens = str.split('=')
        token1 = tokens[0]
        token2 = tokens[1][1:-1]

        sub_tokens_1 = cut_hump(token1).split(' ')
        sub_tokens_2 = cut_hump(token2).split(' ')

        for sub_token in sub_tokens_1:
            result.append(sub_token)
        result.append('=')
        for sub_token in sub_tokens_2:
            result.append(sub_token)

    else:
        sub_tokens = cut_hump(str).split(' ')
        for sub_token in sub_tokens:
            result.append(sub_token)

    return result


def cut_hump(str):
    result = []
    allLowerRegex = "[a-z]+\Z"
    allUpperRegex = "[A-Z]+\Z"
    numRegex = "[0-9]+\Z"
    capitalRegex = "[A-Z][a-z]+\Z"

    if str.find('_') != -1:
        tokens = str.split('_')
        for token in tokens:
            result.append(cut_hump(token))
            result.append(" ")
        result.pop()
    elif re.match(allLowerRegex, str) or re.match(numRegex, str):
        return str
    elif re.match(allUpperRegex, str) or re.match(capitalRegex, str):
        return str.lower()
    else:
        n = len(str)
        flag = False
        for i in range(n):
            c = str[i]
            if (c.isupper()):
                if len(result) == 0:
                    result.append(c.lower())
                elif not flag:
                    result.append(' ')
                    result.append(c.lower())
                else:
                    if i == n - 1:
                        result.append(c.lower())
                    else:
                        next_c = str[i + 1]
                        if next_c.islower():
                            result.append(' ')
                            result.append(c.lower())
                        else:
                            result.append(c.lower())
                flag = True
            elif c.islower():
                if len(result) != 0 and result[-1].isdecimal():
                    result.append(' ')
                    result.append(c)
                else:
                    result.append(c)
                flag = False
            elif (c.isdecimal()):
                if len(result) != 0 and result[-1].isdecimal():
                    result.append(c)
                else:
                    result.append(' ')
                    result.append(c)
                flag = False

    return "".join(result)


def float_to_percent(num:float) -> str:
    """
    浮点到百分比表示 保留两位小数
    :param num: 要转换的浮点数
    :return: 百分比表示
    """
    return "%.2f%%" % (num * 100)


def random_unit(p: float):
    """
    以p概率执行某段函数
    :param p:
    :return:
    """
    R = random()
    if R < p:
        return True
    else:
        return False
