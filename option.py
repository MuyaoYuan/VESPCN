import argparse
import template

parser = argparse.ArgumentParser(description='SR')

parser.add_argument('-tp','--template', default='.',
                    help='You can set various templates in template.py')
parser.add_argument('-t','--task', default='.',
                    help='preparation, train, reload-pre or reload-trained')

parser.add_argument('--num_workers', default=8,
                    help='the num of workers')
parser.add_argument('--pin_memory', default=True,
                    help='pin_memory')

args = parser.parse_args()
template.set_template(args)