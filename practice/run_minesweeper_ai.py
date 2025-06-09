# -*- coding: utf-8 -*-
import os
import sys
import tkinter as tk
from minesweeper_ai import MinesweeperAI

def main():
    """지뢰찾기 AI 실행 스크립트"""
    print("=" * 50)
    print("지뢰찾기 AI - 딥러닝 기반 자동 플레이")
    print("=" * 50)

    # 난이도 선택
    print("\n난이도를 선택하세요:")
    print("1. 초급 (9x9, 지뢰 10개)")
    print("2. 중급 (16x16, 지뢰 40개)")
    print("3. 고급 (30x16, 지뢰 99개)")
    print("4. 커스텀 설정")

    difficulty = input("\n선택 (기본: 1): ") or "1"

    # 난이도에 따른 설정
    if difficulty == "1":
        width, height, mines = 9, 9, 10
    elif difficulty == "2":
        width, height, mines = 16, 16, 40
    elif difficulty == "3":
        width, height, mines = 30, 16, 99
    elif difficulty == "4":
        # 커스텀 설정
        width = int(input("가로 크기 (8-30): ") or "9")
        height = int(input("세로 크기 (8-24): ") or "9")
        max_mines = (width * height) - 9  # 최소 9칸은 비워둠
        mines = int(input(f"지뢰 수 (1-{max_mines}): ") or "10")

        # 유효성 검사
        width = max(8, min(width, 30))
        height = max(8, min(height, 24))
        mines = max(1, min(mines, max_mines))
    else:
        width, height, mines = 9, 9, 10

    print(f"\n선택된 난이도: {width}x{height}, 지뢰 {mines}개")

    # AI 에이전트 생성
    ai = MinesweeperAI(width, height, mines)

    # 모델 파일 경로
    model_path = f"minesweeper_model_{width}x{height}_{mines}.h5"
    model_exists = os.path.exists(model_path)

    # 모드 선택
    print("\n모드를 선택하세요:")
    print("1. AI 학습 시작")
    print("2. 학습된 AI로 게임 플레이" + (" (학습된 모델 있음)" if model_exists else " (학습된 모델 없음)"))

    mode = input("\n선택 (기본: 1): ") or "1"

    if mode == "1":
        # 학습 설정
        episodes = int(input("\n학습 에피소드 수 (기본: 1000): ") or "1000")
        visualize = input("학습 과정 시각화? (y/n, 기본: n): ").lower() == 'y'

        # 학습 시작
        print(f"\n학습 시작: {episodes} 에피소드")
        print("학습 중... (종료하려면 Ctrl+C)")

        try:
            ai.train(episodes=episodes, visualize=visualize)
            print("\n학습 완료!")

            # 학습 후 게임 플레이
            play_after_train = input("\n학습 후 게임 플레이? (y/n, 기본: y): ").lower() != 'n'
            if play_after_train:
                print("\nAI가 게임을 플레이합니다...")
                ai.play_game()
        except KeyboardInterrupt:
            print("\n\n학습이 중단되었습니다.")
            # 중간에 중단되어도 모델 저장
            ai.agent.save(model_path)
            print(f"현재까지의 모델이 저장되었습니다: {model_path}")

            # 중단 후 게임 플레이
            play_after_interrupt = input("\n중단 후 게임 플레이? (y/n, 기본: n): ").lower() == 'y'
            if play_after_interrupt:
                print("\nAI가 게임을 플레이합니다...")
                ai.play_game()
    else:
        # 학습된 모델로 게임 플레이
        if not model_exists:
            print("\n경고: 학습된 모델이 없습니다. 랜덤하게 플레이합니다.")

        print("\nAI가 게임을 플레이합니다...")
        ai.play_game()

    print("\n프로그램 종료")

if __name__ == "__main__":
    main()
