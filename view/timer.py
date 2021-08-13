from multiprocessing import Process

import PySimpleGUI as sg
import time


# Based on the timer example provided by PySimpleGUI

def _start_timer():
    sg.theme('DarkGrey6')

    layout = [[sg.Text('Waiting for results...', font=('Helvetica', 20))],
              [sg.Text('', size=(8, 2), font=('Helvetica', 20),
                       justification='center', key='text')]]

    window = sg.Window('Running Timer', layout,
                       auto_size_buttons=False,
                       keep_on_top=True,
                       grab_anywhere=True,
                       element_padding=(0, 0),
                       finalize=True,
                       element_justification='c',
                       right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_EXIT)

    def time_as_int():
        return int(round(time.time() * 100))

    current_time, paused_time, paused = 0, 0, False
    start_time = time_as_int()

    while True:
        if not paused:
            event, values = window.read(timeout=10)
            current_time = time_as_int() - start_time
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
        else:
            event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == '-RESET-':
            paused_time = start_time = time_as_int()
            current_time = 0
        elif event == '-RUN-PAUSE-':
            paused = not paused
            if paused:
                paused_time = time_as_int()
            else:
                start_time = start_time + time_as_int() - paused_time
            # Change button's text
            window['-RUN-PAUSE-'].update('Run' if paused else 'Pause')
        elif event == 'Edit Me':
            sg.execute_editor(__file__)
        # --------- Display timer in window --------
        window['text'].update('{:02d}:{:02d}.{:02d}'.format((current_time // 100) // 60,
                                                            (current_time // 100) % 60,
                                                            current_time % 100))


def open_timer() -> Process:
    """
    Opens timer screen
    """
    p = Process(target=_start_timer)
    p.start()
    return p
