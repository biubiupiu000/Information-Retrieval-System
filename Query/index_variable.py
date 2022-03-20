import re


r_list = ['.', ',', '/', '?', '!', '%', '_', ';', '*', '&', '-', '#', '$', '×', '§', '^', ' ']
#common list for prefix
pre_list = ['anti', 'dis', 'en', 'em', 'fore', 'co', 'ex', 'ir', 'inter', 'mid', 'mis', 'post', 'over', 'pre', 're',
            'semi', 'sub', 'super', 'trans', 'un', 'under', 'self']
#common list for file extensions
file_ext_list = ['wav', 'wma', 'rar', 'zip', 'csv', 'tar', 'xml', 'email', 'apk', 'exe', 'jpeg', 'png',
                 'ico', 'html', 'php', 'pdf', 'xls', 'xlsx', 'doc', 'txt']

date_map = {'January': '01', 'February': '02', "March": '03', 'April': '04', 'May': '05', 'June': '06',
            'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'}
regex1 = re.compile(r'^<.*->') # remove lines sort of '<***->'
regex2 = re.compile(r'^<[/A-Z]*>$') # remove tags
sp_regex1 = re.compile(r'(?:[a-zA-Z]{1,}\.)+\b[a-zA-Z]*\b') # acronyms
sp_regex2 = re.compile(r'[€£\$¥§] \d+') # currency
sp_regex3 = re.compile(r'[a-zA-Z]+-\d+') # alpha-digit
sp_regex4 = re.compile(r'\d+-[a-zA-Z]+') # digit-alpha
sp_regex5 = re.compile(r'(?:[a-zA-Z]+-)+[a-zA-Z]*') # hyphen-word
sp_regex6 = re.compile(r'[0-9a-zA-Z_\.]{0,19}@[0-9a-zA-Z]{1,13}\.(?:com|cn|gov|net|edu)') # email
sp_regex7 = re.compile(r'(((25[0-5]|2[0-4]\d|1\d{2})|([1-9]?\d))\.){3}((25[0-5]|2[0-4]\d|1\d{2})|([1-9]?\d))') # ip-address
sp_regex8 = re.compile(r'(?:https?://)?[-A-Za-z0-9+&@#/%?=~_|!:,\.;]+\.(?:net|gov|edu|com|cn)') # url
sp_regex9 = re.compile(r'[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?') # digit
a = re.compile(
    r'(:?January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s\d{1,2},\s\d{4}')# unify date