from data.Data import radio128, radio512, radio1024, all_radio128, all_radio512

def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    if args.set == 'radio128':
        dataset = radio128()
    elif args.set == 'radio512':
        dataset = radio512()
    elif args.set == 'radio1024':
        dataset = radio1024()
    elif args.set == 'all_radio128':
        dataset = all_radio128()
    elif args.set == 'all_radio512':
        dataset = all_radio512()
    return dataset