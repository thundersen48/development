import math
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt
import time 


ANIMATION_DT = 0.01  # Скорость самой анимации
CONVEYOR_SPEED = 0.005 # Скорость движения коробок по конвейеру


BASE_POSE = SE3(0, 0, 0)
CONVEYOR = SE3(0.4, 0.3, 0.0)
PALLET = SE3(0.4, -0.3, 0.0)

# Размеры 
CONVEYOR_WIDTH = 0.6
CONVEYOR_DEPTH = 0.2
PALLET_WIDTH = 0.2
PALLET_DEPTH = 0.2


Z_SAFE = 0.35       # безопасная выоста пролета над которой робот перемещается в м. 
Z_PICK = 0.02       # Насколько низко опускается захват относительно поверхности

# Начальные относительные координаты коробок на конвейере.
BOX_OFFSETS_CONVEYOR_START = [
    SE3(-0.25, 0, 0), 
    SE3(-0.15, 0, 0),
    SE3(-0.05, 0, 0),
    SE3( 0.05, 0, 0)  # Ближайшая к роботу коробка
]

# Целевые относительные позиции коробок на паллете 
BOX_OFFSETS_PALLET = [
    SE3(-0.05, -0.05, 0),
    SE3( 0.05, -0.05, 0),
    SE3(-0.05,  0.05, 0),
    SE3( 0.05,  0.05, 0)
]


def create_robot():
    """
    a - длина звена вдоль xi 
    alpa - угол скручивания вокруг xi
    d - смещение звена вдоль zi-1
    theta - угол вокруг оси zi-1, по умолчанию используется q[0]
    """
    L1 = rtb.RevoluteDH(d=0.20, a = 0, alpha=math.pi/2)   # Основание d - высота основания робота, alpha = pi/2, чтобы ось Z1 была перпед-на Z0
    L2 = rtb.RevoluteDH(a=0.35, alpha=0)  # a - длина саомого звена              
    L3 = rtb.RevoluteDH(a=0.30, alpha=0)  # a - длина саомого звена            
    L4 = rtb.RevoluteDH(a=0.00, alpha=-math.pi/2)       
    L5 = rtb.RevoluteDH(d=0.15, alpha=0)# d - вылет захвата по оси z, расстояние от последней оси до точки на конце инструмента               

    robot = rtb.DHRobot([L1, L2, L3, L4, L5], name="Manipualtor")
    robot.base = BASE_POSE
    
    # Стартовая поза
    robot.q = np.array([0.0, -0.5, 1.0, 0.0, 0.0])
    return robot


def get_ik(robot, T_target, q_seed=None):
    """
    T - traget - целевая матрица
    q_seed - нач. приближение
    Функция находит набор углов q, который переместит захват робота в желаемое положение и ориентацию.
    Если первая попытка не удается, она пробует другие начальные предположения
    """
    if q_seed is None: q_seed = robot.q # 
    
    # 1 Попытка - От текущего положения
    sol = robot.ikine_LM(T_target, q0=q_seed, ilimit=300, tol=1e-5) 
    if sol.success: return sol.q 
    """ 
    алгоритм обратной кинематики Levenberg-Marquardt
    limit - уст макс кол0во итераций
    tol - устанавливает допуск
    Если решение найдено, функция  возвращает найденные углы сочленений sol.q
    """ 

    # 2 Попытка - локтем вверх 
    q_crane = np.array([0.0, -0.5, 1.5, -1.0, 0.0]) # зараенее опредленый набор углов
    sol = robot.ikine_LM(T_target, q0=q_crane, ilimit=500, tol=1e-5)
    if sol.success: return sol.q

    # 3 Попытка - Все сочленения устанавливаются в нулевое положение
    sol = robot.ikine_LM(T_target, q0=np.zeros(5), ilimit=500, tol=1e-5)
    if sol.success: return sol.q

    return None


def generate_trajectory(robot, start_q, end_q, steps):
    """
    start_q - Вектор углов сочленений начальной позы
    end_q - Вектор углов сочленений конечной позы
    steps - желаемое кол-во шагов между начальной и конченой позой
    Генерирует траекторию между двумя позами, путем вычисления последовательности промежуточных углов для каждого сочленения,
    обеспечивая плавный переход. Возвращает массив углов q для каждого steps. Массив содержит steps строк, где каждая строка - поза робота(5 углов) 
    """
    return rtb.jtraj(start_q, end_q, steps).q


def main():
    """
    Выполняется инициализация; Переход в зону ожидания; цикл обработки коробок и визуализация
    """

    robot = create_robot()
    
    """
    Здесь задаются фиксированные мировые координаты для укладки на паллет и начальные координаты коробок на конвейере
    offset - матрица, определяющая относительное смещение одной коробки внутри паллета или конвейера 
    """
    target_pallet_poses = [PALLET * offset for offset in BOX_OFFSETS_PALLET]

    # Список текущих позиций коробок на конвейере
    current_conveyor_boxes = [CONVEYOR * offset for offset in BOX_OFFSETS_CONVEYOR_START]

    # Безопасная домашняя точка ожидания робота
    q_home = np.array([0.0, -0.4, 1.0, -0.5, 0.0])
    current_q = q_home.copy()

   
    timeline = [] # Список для хранения всей последовательности движений (q, carry_index, conveyor_box_states)

    # Начальная траектория
    """
    Робот плавно переходит из своей стартовой позы (robot.q) в домашнюю (q_home), при этом никаких коробок он не несет (None)
    """
    traj = generate_trajectory(robot, robot.q, q_home, 20)
    for q_step in traj:
        timeline.append((q_step, None, current_conveyor_boxes.copy()))

    "Цикл обработки коробок"
    for i in range(4): 
        print(f"Расчет коробки {i+1}/4...")
        

        # Вычисляем конечную точку на конвейере для захвата
        pick_point_on_conveyor = CONVEYOR * SE3(BOX_OFFSETS_CONVEYOR_START[i].t[0], 0, 0) # X-координата первой коробки, Y и Z - центр конвейера
        
        # Перемещаем все коробки, которые еще на конвейере
        
        # Расчет точек захвата и укладки
        T_pick_high = pick_point_on_conveyor * SE3(0, 0, Z_SAFE)
        T_pick_low  = pick_point_on_conveyor * SE3(0, 0, Z_PICK)
        T_place_high = target_pallet_poses[i] * SE3(0, 0, Z_SAFE)
        T_place_low  = target_pallet_poses[i] * SE3(0, 0, Z_PICK)

        # Вызывается функция обратной кинематики 
        q_pick_h = get_ik(robot, T_pick_high, q_seed=current_q)
        if q_pick_h is None: q_pick_h = current_q
        
        q_pick_l = get_ik(robot, T_pick_low, q_seed=q_pick_h)
        if q_pick_l is None: q_pick_l = q_pick_h

        q_place_h = get_ik(robot, T_place_high, q_seed=q_home) # Из домашней позиции к месту укладки
        if q_place_h is None: q_place_h = q_home

        q_place_l = get_ik(robot, T_place_low, q_seed=q_place_h)
        if q_place_l is None: q_place_l = q_place_h

        "Генерация и запись траекторий"
        # 1. К коробке(захват)
        path_pick = [
            (current_q, q_pick_h, 15, None),
            (q_pick_h,  q_pick_l, 10, None),
            (q_pick_l,  q_pick_h, 10, i) # Индекс i означает, что робот несет i-ю коробку
        ]
        
        # 2. Через домашней позиции к паллете 
        path_transit = [
            (q_pick_h, q_home, 15, i),
            (q_home,   q_place_h, 15, i)
        ]

        # 3. Укладка
        path_place = [
            (q_place_h, q_place_l, 10, i),
            (q_place_l, q_place_h, 10, None) # Коробка положена (None)
        ]

        full_path = path_pick + path_transit + path_place

        for q_start, q_end, steps, carry in full_path:
            traj_segment = generate_trajectory(robot, q_start, q_end, steps)
            for q_step in traj_segment:
                timeline.append((q_step, carry, current_conveyor_boxes.copy())) # Все шаги записываем в timeline    
        """
        Текущее положение робота обновляется для следующей итерации цикла. 
        Следующая коробка будет захватываться, начиная поиск ik из этой последней известной позиции.
        """
        current_q = q_place_h.copy() 
        
        # Удаляем коробку С конвейера после захвата
        # Мы предполагаем, что i-я коробка - это та, которую мы взяли.
        # Удаляем ее из списка "на конвейере"
        if len(current_conveyor_boxes) > 0:
            current_conveyor_boxes.pop(0) # Удаляем первую коробку, как будто она взята

    # Возвращение в дом. позицию
    traj_final_home = generate_trajectory(robot, current_q, q_home, 20)
    for q_step in traj_final_home:
        timeline.append((q_step, None, current_conveyor_boxes.copy()))

    print(f"Расчет окончен. Всего кадров: {len(timeline)}")

   

    # Визуализация
    env = robot.plot(robot.q, backend='pyplot', block=False, eeframe=False, jointaxes=False)
    ax = env.ax
    

    ax.set_xlim(-0.5, 0.8)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(0.0, 0.8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Робот-манипулятор")
    

    def plot_rect(center, w, d, c, label=None):
        x = [center.t[0] - w/2, center.t[0] + w/2, center.t[0] + w/2, center.t[0] - w/2, center.t[0] - w/2]
        y = [center.t[1] - d/2, center.t[1] - d/2, center.t[1] + d/2, center.t[1] + d/2, center.t[1] - d/2]
        ax.plot(x, y, [0]*5, color=c, linewidth=2, label=label)
    
    plot_rect(CONVEYOR, CONVEYOR_WIDTH, CONVEYOR_DEPTH, "#363636", 'Конвейер')
    plot_rect(PALLET, PALLET_WIDTH, PALLET_DEPTH, "#826107", 'Паллет')
    ax.legend()

    # Инициализация всех коробок
    # 4 коробки, которые будут перемещаться по конвейеру, а потом браться
    conveyor_box_visuals = []
    for _ in range(len(BOX_OFFSETS_CONVEYOR_START)):
        # Создаем невидимые, они появятся когда их возьмут
        b = ax.scatter([0], [0], [0], c="#EA8603", s=100, visible=False)
        conveyor_box_visuals.append(b)

    # 4 коробки, которые будут уложены на паллету
    pallet_box_visuals = []
    for _ in range(len(BOX_OFFSETS_PALLET)):
        b = ax.scatter([0], [0], [0], c='red', marker='x', s=100, visible=False)
        pallet_box_visuals.append(b)
    
    #отслеживание, какая коробка сейчас в захвате
    current_held_box_visual = ax.scatter([0],[0],[0], c='magenta', s=150, visible=False)

    # --- ЦИКЛ АНИМАЦИИ ---
    for step_idx, (q, carry_idx, conveyor_states) in enumerate(timeline):
        robot.q = q
        env.step(ANIMATION_DT) # Обновляет робота
        
        # --- Обновление коробок на конвейере ---
        # conveyor_states - это список объектов SE3, которые еще на конвейере
        # i-тая коробка на конвейере это conveyor_states[i]
        
        for idx, box_pose in enumerate(conveyor_states):
            if idx < len(conveyor_box_visuals): # Убеждаемся, что есть объект для отрисовки
                box_vis = conveyor_box_visuals[idx]
                box_vis.set_visible(True)
                box_vis._offsets3d = ([box_pose.t[0]], [box_pose.t[1]], [box_pose.t[2] + Z_PICK])
            else:
                if idx < len(conveyor_box_visuals):
                    conveyor_box_visuals[idx].set_visible(False)

        # Скрываем лишние, если коробок стало меньше
        for idx in range(len(conveyor_states), len(conveyor_box_visuals)):
            conveyor_box_visuals[idx].set_visible(False)

        # --- Обновление коробки в руке робота ---
        if carry_idx is not None:
            T_hand = robot.fkine(q)
            new_pos = (T_hand.t[0], T_hand.t[1], T_hand.t[2] - 0.05)
            current_held_box_visual.set_visible(True)
            current_held_box_visual._offsets3d = ([new_pos[0]], [new_pos[1]], [new_pos[2]])
        else:
            current_held_box_visual.set_visible(False) # Робот ничего не несет

        # --- Обновление уложенных коробок на паллете ---
        # Если carry_idx == None И текущий шаг - это шаг после укладки i-той коробки
        # Мы должны показать i-тую коробку на паллете.

        
        num_placed_boxes = 0
        # Проходим по timeline до текущего шага и считаем, сколько раз carry_idx менялся с i на None
        for prev_q, prev_carry, _ in timeline[:step_idx + 1]:
            if prev_carry is not None and prev_carry < len(BOX_OFFSETS_PALLET):
                # Проверяем, была ли эта коробка только что отпущена
                if step_idx > 0 and timeline[step_idx-1][1] == prev_carry and prev_carry is not None and carry_idx is None:
                     num_placed_boxes = prev_carry + 1 # Если i-я коробка отпущена, то num_placed_boxes = i+1
                     break

        completed_box_cycles = 0
        current_processing_box_idx = -1
        for s_idx in range(step_idx + 1):
            _q, _carry_idx, _conveyor_states = timeline[s_idx]
            if _carry_idx is not None:
                current_processing_box_idx = _carry_idx
            elif current_processing_box_idx != -1 and _carry_idx is None:
                # Коробка отпущена
                if current_processing_box_idx == completed_box_cycles:
                    completed_box_cycles += 1
                current_processing_box_idx = -1

        for box_idx in range(completed_box_cycles):
            if box_idx < len(pallet_box_visuals):
                target_pose = target_pallet_poses[box_idx]
                pallet_box_visuals[box_idx].set_visible(True)
                pallet_box_visuals[box_idx]._offsets3d = ([target_pose.t[0]], [target_pose.t[1]], [target_pose.t[2] + Z_PICK])
        
        # Скрываем те, что не должны быть видны
        for box_idx in range(completed_box_cycles, len(pallet_box_visuals)):
            pallet_box_visuals[box_idx].set_visible(False)


    plt.show(block=True)

if __name__ == "__main__":
    main()