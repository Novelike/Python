# 기초 249p
# • 여러 개의 숫자를 입력받아 합계를 계산하는 함수를 작성
# • 사용자가 'q'를 입력하면 입력을 중단하고 지금까지 입력한 숫자의 합을 출력
def sum_numbers_with_expression():
    input_numbers = []
    
    while True:
        user_input = input("숫자를 입력하세요 (종료하려면 'q' 입력): ")
        
        if user_input.lower() == 'q':
            break
        
        try:
            number = int(user_input)
            input_numbers.append(number)
            
            # 계산식 표시
            calculation_expression = " + ".join(str(num) for num in input_numbers)
            print(f"현재 계산식: {calculation_expression}")
            
        except ValueError:
            print("유효한 숫자가 아닙니다. 다시 시도하세요.")
    
    calculated_sum = sum(input_numbers)
    return calculated_sum, input_numbers

# 함수 실행 및 결과 출력
if __name__ == "__main__":
    total, numbers = sum_numbers_with_expression()
    
    if numbers:
        expression = " + ".join(str(num) for num in numbers)
        print(f"최종 계산식: {expression} = {total}")
    else:
        print("입력된 숫자가 없습니다.")