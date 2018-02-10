def write_to_file(prefix, map_nums):
    with open('{}_maps.txt'.format(prefix), 'w') as f:
        f.write(','.join('training-09x09-{:04d}'.format(map_num)
                         for map_num in map_nums))


def generate_map_list(num, exclude_maps=frozenset()):
    return [i for i in xrange(1, num + 1) if i not in exclude_maps]


def main():
    static_maps = [127, 169, 246, 336, 445, 589, 691, 828, 844, 956]
    exclude_maps = frozenset([2])

    write_to_file('static', static_maps)
    write_to_file('1', [1])
    write_to_file('10', generate_map_list(10, exclude_maps))
    write_to_file('100', generate_map_list(100, exclude_maps))
    write_to_file('500', generate_map_list(500, exclude_maps))
    write_to_file('1000', generate_map_list(1000, exclude_maps))


if __name__ == '__main__':
    main()
