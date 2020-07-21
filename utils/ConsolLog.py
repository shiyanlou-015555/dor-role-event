def Print(msg, msg_type='success'):
    """
    Colorful output.

    Args:
        msg_type: Valid in range ['error', 'success', 'warning', 'information', 'normal']
    """
    if msg == '':
        Print('Empty message to log.', 'warning')
        return
    if msg_type is 'error':
        msg = '\033[31m' + msg + '\033[0m'
    if msg_type is 'success':
        msg = '\033[32m' + msg + '\033[0m'
    if msg_type is 'warning':
        msg = '\033[33m' + msg + '\033[0m'
    if msg_type is 'information':
        msg = '\033[34m' + msg + '\033[0m'
    else:
        msg = '\033[37m' + msg + '\033[0m'
    print(msg)
