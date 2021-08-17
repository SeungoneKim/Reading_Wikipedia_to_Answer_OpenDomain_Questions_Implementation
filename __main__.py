if __name__ == "__main__":

    sys.stdout.write('#################################################\n')
    sys.stdout.write('You have entered __main__.\n')
    sys.stdout.write('#################################################\n')

    sys.stdout.write('#################################################\n')
    context = open('context.txt','r')
    print('The given context is : ')
    print()
    print(context)
    sys.stdout.write('#################################################\n')

    sys.stdout.write('#################################################\n')
    query = input('What is your query : ')
    print('The given query is : ')
    print()
    print(query)
    sys.stdout.write('#################################################\n')

    sys.stdout.write('#################################################\n')
    model = input('Which model would you use(BERT / RoBERTa) : ')
    sys.stdout.write('#################################################\n')

    if model == 'BERT':
        Key_Information_Extraction_BERT(context, query, False)
    elif model == 'RoBERTa':
        Key_Information_Extraction_RoBERTa(context, query, False)
    else:
        assert "Model is not supported yet!"

    sys.stdout.write('#################################################\n')
    sys.stdout.write('You are exiting __main__.\n')
    sys.stdout.write('#################################################\n')
