class RTSNet(nn.Module):
    def __init__(self, p=0.3, in_dim=6, out_dim=3, n_layers=1):
        super(RTSNet, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim*4),
            nn.BatchNorm1d(in_dim*4),
            nn.GELU(),
            nn.Dropout(p),
        )

        layers = []
        dim = in_dim * 4
        for i in range(n_layers):
          layers.extend([
            nn.Linear(in_features=dim, out_features=in_dim*4),
            nn.BatchNorm1d(in_dim*4),
            nn.GELU(),
            nn.Dropout(p),
          ])
        
        self.layers = nn.Sequential(*layers)

        self.last_layer = nn.Linear(in_dim*4, out_dim)
        
    def forward(self, x):
        x = self.first_layer(x)
        
        x = self.layers(x)

        return self.last_layer(x)
