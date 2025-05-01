import gymnasium as gym
from pynput import keyboard
import time

def landing():
    """
    pynput 实时控制 LunarLander-v3
    ←  : action 1  (左辅推)
    ↑  : action 2  (主推)
    →  : action 3  (右辅推)
    Esc: 退出
    """

    # ---------- 1. 键盘监听 ----------
    current_action = 0           # 0-3
    _pressed = set()             # 记录正在按下的方向键名

    def _update_action():
        nonlocal current_action
        # 定义优先级：左 > 上 > 右，可按需调整
        if "up" in _pressed:
            current_action = 2
        elif "left" in _pressed:
            current_action = 1
        elif "right" in _pressed:
            current_action = 3
        else:
            current_action = 0

    def _on_press(key):
        if key == keyboard.Key.esc:
            return False         # 停止 listener => 抛出异常结束主循环
        if key == keyboard.Key.left:   _pressed.add("left")
        if key == keyboard.Key.up:     _pressed.add("up")
        if key == keyboard.Key.right:  _pressed.add("right")
        _update_action()

    def _on_release(key):
        if key == keyboard.Key.left:   _pressed.discard("left")
        if key == keyboard.Key.up:     _pressed.discard("up")
        if key == keyboard.Key.right:  _pressed.discard("right")
        _update_action()

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)    # type: ignore
    listener.start()               # 后台线程，立即生效

    # ---------- 2. 创建环境 ----------
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = env.reset()

    def print_info(obs, reward, terminated, truncated, info):
        """
        打印信息
        """
        print("-" * 20)
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        print(f"info: {info}")
        print(f"current_action: {current_action}")
        print("-" * 20)

    # ---------- 3. 主循环 ----------
    try:
        done = False
        step = 0
        while not done:
            step += 1
            obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated
            env.render()

            print_info(obs, reward, terminated, truncated, info)

            time.sleep(1 / env.metadata["render_fps"])
    except (KeyboardInterrupt, StopIteration):
        # listener 被 Esc 停止或 Ctrl+C
        pass
    finally:
        env.close()
        listener.stop()
