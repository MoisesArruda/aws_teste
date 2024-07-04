# IA Generativa na AWS

### Bedrock

* Recurso principal da AWS para se trabalhar com IA Generativa, é de lá que iremos consumir a API dos modelos(o que eles chamam de Foundation Models) para utilizar em nossas aplicações.

    **Knowledge Bases** - Esta é a funcionalidade do BedRock onde iremos conectar nosso bucket do S3 com nossos modelos, para que sejá possível criar a base de conhecimento para servir como contexto. 

    Para maiores informações favor consultar [este link!](https://github.com/aws-samples/amazon-bedrock-samples/blob/main/knowledge-bases/01-rag-concepts/1a_create_ingest_documents_test_kb.ipynb) 

    **Vector Storage** - Para que os modelos do bedrock possam utilizar as informações dos arquivos como contexto, é necessário passar um recurso para que sirva como armazenamento vertorial. Estes podem ser criados utilizando o Open Search Serveless ou algum outro recurso de sua prefêrencia que servirá como Banco de Dados Vetorial. 

    O modelo utilizado para realizar o word embeddings também deve ser definido neste momento! 

### Open Search:

* Talvez este serviço seja a principal opção de banco/repositório de dados vetorial da AWS, aqui é possível escolher entre as opções Cluster e Serveless

    **Serveless** - Opção mais cara. Nela é possível passar algumas configurações de embeddings, como campo vetorial, processamento etc.

        Index(Índice) - Será o "repositório" dentro do OSS onde os embeddings serão armazenados

    **Cluster** - Mais barato, porém pelo visto não se conecta com o Bedrock. Utiliza o mesmo modelo de Cluster.

### Amazon Kendra

* Este recurso é utilizado para realizar a busca por similaridade, é um conceito diferente da pesquisa vetorial, porém, dependendo da necessidade poderia ser utilizado.