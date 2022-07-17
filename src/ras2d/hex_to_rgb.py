def hex_to_rgb(value: str):
    '''
    Convert a HEC color string to a tuple of RGB color values, 0 - 255
    '''
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    

if __name__ == '__main__':
    colors = ('#101882', '#18318F', '#1A398F', '#2657A0', '#377BB0', '#469CC3', '#57BCD9', '#62D0E6')
    c = [hex_to_rgb(color) for color in colors]
    print(c)