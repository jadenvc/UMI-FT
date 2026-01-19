from colorama import init, Fore, Back, Style
init()

# Dictionary for foreground colors
foreground_colors = {
    'black': Fore.BLACK,
    'red': Fore.RED,
    'green': Fore.GREEN,
    'yellow': Fore.YELLOW,
    'blue': Fore.BLUE,
    'magenta': Fore.MAGENTA,
    'cyan': Fore.CYAN,
    'white': Fore.WHITE,
    'reset': Fore.RESET
}

# Dictionary for background colors
background_colors = {
    'black': Back.BLACK,
    'red': Back.RED,
    'green': Back.GREEN,
    'yellow': Back.YELLOW,
    'blue': Back.BLUE,
    'magenta': Back.MAGENTA,
    'cyan': Back.CYAN,
    'white': Back.WHITE,
    'reset': Back.RESET
}

def color_print(text='', color='magenta', style='fore'):
    assert style in ['fore', 'back'], f"Invalid style: {style}, must be 'fore' or 'back'"
    assert color in foreground_colors.keys() or color in background_colors.keys(), f"Invalid color: {color}"
    
    if style == 'fore':
        color_code = foreground_colors.get(color, Fore.GREEN)
      
    elif style == 'back':
        color_code = background_colors.get(color, Back.GREEN)
    else:
        color_code = Style.RESET_ALL  # Default to reset if style is invalid

    # Print with selected color and reset style after text
    print(color_code + text + Style.RESET_ALL)
    
def debug_print(text = ''):
    color_print('[DEBUG] '+ text, color='red', style='back')
    
def info_print(text = '', color='yellow', style='back'):
    color_print('[INFO] '+ text, color=color, style=style)