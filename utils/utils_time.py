from dateutil.relativedelta import relativedelta

def get_date_relativedelta(
    now_date,
    window_unit='months',
    window=2,
    operate='+'
):
    """
    例子:
    get_date_relativedelta(
    now_date=pd.to_datetime('2007-3-30'),
    window_unit='months',
    window=1,
    operate='-'
    )
    :param now_date:
    :param window_unit:
    :param window:
    :param operate:
    :return:
    """
    command = 'now_date + relativedelta( {window_unit} = {operate}{window} )'.format(
        window_unit=window_unit,
        operate=operate,
        window=window
    )
    new_date = eval(command)
    return new_date



