card = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
order = range(1,14)
card_order = dict(zip(card,order))
card_order['1'] = 1
  
  
def cal_24(cards):
    if 'joker' in cards or 'Joker'in cards:
        return 'ERROR'
    # nums = [card_order[card] for card in cards]
    ans = None
    generate_num_order([], cards)
    for num_order in ordered_num:
        ans = recursion_core(num_order[1:], card_order[num_order[0]])
        if ans:
            ans = str(num_order[0]) + ans
            break
    return ans
  
def recursion_core(res_num, cal):
    cur_num = card_order[res_num[0]]
    cur_card = res_num[0]
    if len(res_num) == 1:
        if cal + cur_num == 24:
           return '+' + cur_card
        elif  cal - cur_num == 24:
           return '-' + cur_card
        elif cal * cur_num == 24:
           return '*' + cur_card
        elif  cal / cur_num == 24:
           return '/' + cur_card
        else:
            return None
    add = recursion_core(res_num[1:], cal+cur_num)
    if add:
        return '+' + cur_card + add
    sub = recursion_core(res_num[1:], cal-cur_num)
    if sub:
        return '-' + cur_card + sub
    mul = recursion_core(res_num[1:], cal*cur_num)
    if mul:
        return '*' + cur_card + mul
    div = recursion_core(res_num[1:], cal/cur_num)
    if div:
        return '/' + cur_card + div
    return None
  
def generate_num_order(nums, res_nums):
    if not res_nums:
        ordered_num.append(nums)
        return
    for n in res_nums:
        new_res_nums = res_nums[:]
        new_res_nums.remove(n)
        generate_num_order(nums + [n], new_res_nums)
  
while True:
    try:
        ordered_num = []
        ans = cal_24(input().split())
        print(ans if ans else 'NONE')
    except:
        break     
