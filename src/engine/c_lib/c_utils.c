int Add(int a, int b)
    {
        return a + b;
    }

double cos_for_LD(int shape_x, int shape_y, double a[shape_x][shape_y], double b[shape_x][shape_y][3])
{
    double res[shape_x][shape_y];
    int i;
    int j;
    for(i = 0; i < shape_x; i++)
    {
        for(j = 0; j < shape_y; j++)
        {
            res[i][j] = a[i][0] * b[i][j][0] + a[i][1] * b[i][j][1] + a[i][2] * b[i][j][2];
        }
    }
    return res[shape_x][shape_y];
}