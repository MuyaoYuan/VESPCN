import argparse
import template

parser = argparse.ArgumentParser(description='SR')

parser.add_argument('--template', default='.',
                    help='You can set various templates in template.py')
parser.add_argument('--task', default='.',
                    help='preparation, train, reload-pre or reload-trained')

args = parser.parse_args()
template.set_template(args)